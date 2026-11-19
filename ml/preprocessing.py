"""
ml/preprocessing.py
===================
Preprocessing bersama untuk TRAINING dan PREDIKSI (Django).
Semua fungsi pembersihan bersifat per-baris (tanpa statistik dari data lain),
sehingga input form web diproses PERSIS sama dengan data latih.
Imputasi, scaling, dan one-hot dilakukan di dalam sklearn Pipeline agar
hanya "belajar" dari data latih (mencegah kebocoran data uji).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "kualitas_kopi.csv"
MODEL_PATH = BASE_DIR / "ml" / "coffee_quality_model.joblib"

TARGET = "Total Cup Points"

# 10 skor sensorik = komponen pembentuk TARGET -> DILARANG jadi fitur (leakage)
LEAKAGE_COLUMNS = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body",
                   "Balance", "Uniformity", "Clean Cup", "Sweetness", "Overall"]

NUMERIC_FEATURES = ["Altitude", "Moisture Percentage",
                    "Category One Defects", "Category Two Defects", "Quakers"]
CATEGORICAL_FEATURES = ["Country of Origin", "Variety", "Processing Method", "Color"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- Aturan normalisasi (keputusan manual berdasarkan nilai unik di CSV) ----
PROCESSING_MAP = {
    "washed / wet": "Washed / Wet", "semi washed": "Washed / Wet",
    "semi-lavado": "Washed / Wet", "double anaerobic washed": "Washed / Wet",
    "natural / dry": "Natural / Dry",
    "double carbonic maceration / natural": "Natural / Dry",
    "pulped natural / honey": "Honey / Pulped Natural",
    "honey,mossto": "Honey / Pulped Natural",
    "wet hulling": "Other", "anaerobico 1000h": "Other",
}
COLOR_MAP = {
    "green": "green", "greenish": "green",
    "bluish-green": "blue-green", "blue-green": "blue-green",
    "yellow-green": "yellow-green", "yellow green": "yellow-green",
    "yellow- green": "yellow-green", "yello-green": "yellow-green",
    "yellowish": "yellowish", "pale yellow": "yellowish",
    "brownish": "brownish", "browish-green": "brownish",
}


def _is_missing(x):
    return x is None or (isinstance(x, float) and np.isnan(x)) or str(x).strip() == ""


def clean_processing(x):
    if _is_missing(x):
        return "unknown"
    return PROCESSING_MAP.get(str(x).strip().lower(), str(x).strip())


def clean_color(x):
    if _is_missing(x):
        return "unknown"
    return COLOR_MAP.get(str(x).strip().lower(), "unknown")


def clean_variety(x):
    """Varietas kosong/unknown -> 'unknown'; campuran (koma, +, &, 'and') -> 'Blend'."""
    if _is_missing(x):
        return "unknown"
    v = str(x).strip()
    if v.lower() in ("unknown", "unknow"):
        return "unknown"
    if re.search(r",|\+|&| and | y ", v, flags=re.I):
        return "Blend"
    return v


def parse_altitude(x):
    """'1700-1930' -> 1815 (rata-rata). Nilai > 3000 diasumsikan FEET -> meter
    (mis. '4895 A 5650'); di luar 100-3000 m setelah konversi -> NaN."""
    if _is_missing(x):
        return np.nan
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", str(x))]
    if not nums:
        return np.nan
    alt = float(np.mean(nums))
    if alt > 3000:
        alt *= 0.3048
    return alt if 100 <= alt <= 3000 else np.nan


def clean_moisture(x):
    """Kadar air 0% tidak realistis -> NaN (akan diimputasi di pipeline)."""
    if _is_missing(x):
        return np.nan
    x = float(x)
    return np.nan if x <= 0 else x


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Terima DataFrame mentah (CSV atau form web), kembalikan fitur bersih."""
    out = pd.DataFrame(index=df.index)
    out["Altitude"] = df["Altitude"].map(parse_altitude)
    out["Moisture Percentage"] = df["Moisture Percentage"].map(clean_moisture)
    for c in ["Category One Defects", "Category Two Defects", "Quakers"]:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["Country of Origin"] = df["Country of Origin"].fillna("unknown").astype(str).str.strip()
    out["Variety"] = df["Variety"].map(clean_variety)
    out["Processing Method"] = df["Processing Method"].map(clean_processing)
    out["Color"] = df["Color"].map(clean_color)
    return out[FEATURES]


def load_dataset(path=DATASET_PATH):
    """Muat CSV -> (X bersih, y). Tidak ada fitur leakage di X."""
    df = pd.read_csv(path)
    X = clean_features(df)
    y = df[TARGET].astype(float)
    assert not set(LEAKAGE_COLUMNS) & set(X.columns), "Leakage terdeteksi!"
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Imputasi + scaling (numerik) dan imputasi + one-hot (kategorikal)."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        # kategori langka (<3 data) digabung; kategori baru saat prediksi tidak error
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=3, sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    X, y = load_dataset()
    print("Shape X:", X.shape, "| y:", y.shape)
    miss = X.isna().sum()
    print("\nMissing per fitur setelah cleaning:\n", miss[miss > 0].to_dict())
    for c in CATEGORICAL_FEATURES[1:]:
        print(f"\n{c}:\n", X[c].value_counts().to_dict())
    print("\nAltitude:\n", X["Altitude"].describe().round(1).to_dict())
    Xtr, Xte, ytr, yte = split_data(X, y)
    Z = build_preprocessor().fit(Xtr).transform(Xte)
    print("\nTrain/Test:", len(Xtr), "/", len(Xte), "| fitur setelah encoding:", Z.shape[1])
    print("NaN setelah preprocessing:", int(np.isnan(Z).sum()))
