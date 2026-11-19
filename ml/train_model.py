"""
ml/train_model.py
=================
Melatih, membandingkan, mengevaluasi, dan menyimpan model Coffee Quality AI.

Jalankan dari folder root proyek:
    python ml/train_model.py

Output:
  - ml/coffee_quality_model.joblib  (pipeline + metadata, dimuat Django tanpa training ulang)
  - static/images/*.png             (grafik EDA & evaluasi untuk dashboard)
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")  # backend tanpa layar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ml.preprocessing import (BASE_DIR, CATEGORICAL_FEATURES, DATASET_PATH, FEATURES,
                              LEAKAGE_COLUMNS, MODEL_PATH, NUMERIC_FEATURES, TARGET,
                              build_preprocessor, load_dataset, split_data)

RANDOM_STATE = 42
IMG_DIR = BASE_DIR / "static" / "images"
BROWN, CREAM, LATTE = "#4b2e1e", "#f5ebe0", "#b08968"


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def make_pipeline(model):
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5,
                                               random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=RANDOM_STATE, n_jobs=-1),
    }


def grouped_importance(pipe):
    """Jumlahkan feature_importances_ hasil one-hot kembali ke fitur asli."""
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    agg = {f: 0.0 for f in FEATURES}
    for n, v in zip(names, imp):
        raw = n.split("__", 1)[1]
        for f in sorted(FEATURES, key=len, reverse=True):
            if raw == f or raw.startswith(f + "_"):
                agg[f] += float(v)
                break
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, name):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG_DIR / name, dpi=130, facecolor="white")
    plt.close(fig)


def make_eda_charts(df_raw, X, y, comparison, imp, y_test, y_pred, best_name):
    # 1. Distribusi Total Cup Points
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(y, bins=20, color=LATTE, edgecolor="white")
    ax.axvline(y.mean(), color=BROWN, ls="--", label=f"Rata-rata {y.mean():.2f}")
    ax.axvline(85, color="#c0392b", ls=":", label="Batas 85 (Excellent)")
    ax.set(title="Distribusi Total Cup Points", xlabel="Total Cup Points", ylabel="Jumlah sampel")
    ax.legend(); style_ax(ax); save(fig, "dist_total_cup_points.png")

    # 2. Distribusi kategori kualitas
    cats = pd.cut(y, [0, 80, 85, 101], right=False,
                  labels=["< 80", "80 - 84,99 (Very Good)", ">= 85 (Excellent)"]).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(cats.index.astype(str), cats.values, color=[CREAM, LATTE, BROWN], edgecolor=BROWN)
    for i, v in enumerate(cats.values):
        ax.text(i, v + 1, str(v), ha="center")
    ax.set(title="Distribusi Kategori Kualitas Kopi", ylabel="Jumlah sampel")
    style_ax(ax); save(fig, "dist_kategori.png")

    # 3. Korelasi: skor sensorik vs target (penjelasan leakage)
    sens = df_raw[LEAKAGE_COLUMNS + [TARGET]].loc[:, lambda d: d.std() > 0]
    corr_s = sens.corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(corr_s.index, corr_s.values, color=LATTE)
    ax.set(title="Korelasi skor sensorik dengan Total Cup Points\n(TIDAK dipakai sebagai fitur: leakage)",
           xlabel="Korelasi Pearson")
    style_ax(ax); save(fig, "corr_sensory_leakage.png")

    # 4. Korelasi fitur numerik model + target
    num = X[NUMERIC_FEATURES].copy(); num[TARGET] = y.values
    c = num.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(c, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(c.columns, fontsize=8)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04); ax.set_title("Korelasi fitur numerik model")
    save(fig, "corr_numeric_features.png")

    # 5. Rata-rata skor per negara (n >= 5)
    g = pd.DataFrame({"c": X["Country of Origin"].values, "y": y.values}).groupby("c")["y"].agg(["mean", "count"])
    g = g[g["count"] >= 5].sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([f"{i} (n={int(n)})" for i, n in zip(g.index, g["count"])], g["mean"], color=LATTE)
    ax.set_xlim(g["mean"].min() - 1, g["mean"].max() + 0.5)
    ax.set(title="Rata-rata Total Cup Points per negara (n >= 5)", xlabel="Rata-rata skor")
    style_ax(ax); save(fig, "mean_by_country.png")

    # 6. Boxplot per metode pengolahan
    order = [k for k in ["Washed / Wet", "Natural / Dry", "Honey / Pulped Natural"] if k in set(X["Processing Method"])]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot([y[X["Processing Method"].values == k] for k in order], tick_labels=order,
               patch_artist=True, boxprops=dict(facecolor=CREAM, color=BROWN), medianprops=dict(color=BROWN))
    ax.set(title="Total Cup Points per metode pengolahan", ylabel="Total Cup Points")
    style_ax(ax); save(fig, "box_processing.png")

    # 7. Perbandingan model (CV RMSE & test R2)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(comparison)
    ax.bar(names, [comparison[n]["cv_RMSE"] for n in names], color=[BROWN if n == best_name else LATTE for n in names])
    ax.axhline(y.std(), color="#c0392b", ls=":", label=f"Baseline (std y = {y.std():.2f})")
    ax.set(title="Perbandingan model: RMSE cross-validation (lebih kecil lebih baik)", ylabel="RMSE")
    ax.legend(); style_ax(ax); save(fig, "model_comparison.png")

    # 8. Feature importance
    if imp:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        k = list(imp)[::-1]
        ax.barh(k, [imp[i] for i in k], color=BROWN)
        ax.set(title=f"Feature importance ({best_name})", xlabel="Kontribusi relatif")
        style_ax(ax); save(fig, "feature_importance.png")

    # 9. Aktual vs prediksi
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_test, y_pred, color=BROWN, alpha=.75)
    lo, hi = min(y_test.min(), y_pred.min()) - .5, max(y_test.max(), y_pred.max()) + .5
    ax.plot([lo, hi], [lo, hi], color="#c0392b", ls="--", label="Prediksi sempurna")
    ax.set(title="Data uji: aktual vs prediksi", xlabel="Aktual", ylabel="Prediksi")
    ax.legend(); style_ax(ax); save(fig, "actual_vs_pred.png")


def main():
    df_raw = pd.read_csv(DATASET_PATH)
    X, y = load_dataset()
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Data latih: {len(X_tr)} | Data uji: {len(X_te)} | Fitur: {FEATURES}")

    # --- Cross-validation HANYA pada data latih ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparison = {}
    for name, model in candidate_models().items():
        cv = cross_validate(make_pipeline(model), X_tr, y_tr, cv=kf,
                            scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"))
        comparison[name] = {"cv_MAE": float(-cv["test_neg_mean_absolute_error"].mean()),
                            "cv_RMSE": float(-cv["test_neg_root_mean_squared_error"].mean()),
                            "cv_R2": float(cv["test_r2"].mean())}
        print(f"[CV] {name:18s} MAE={comparison[name]['cv_MAE']:.3f} "
              f"RMSE={comparison[name]['cv_RMSE']:.3f} R2={comparison[name]['cv_R2']:.3f}")

    # Pemenang = RMSE CV terendah
    best_name = min(comparison, key=lambda n: comparison[n]["cv_RMSE"])
    print("Model terbaik (CV RMSE):", best_name)

    # --- Evaluasi test set untuk semua model (pelaporan), model terbaik di-fit ulang ---
    test_results = {}
    fitted = {}
    for name, model in candidate_models().items():
        p = make_pipeline(model).fit(X_tr, y_tr)
        fitted[name] = p
        test_results[name] = metrics(y_te, p.predict(X_te))
    baseline = metrics(y_te, np.full(len(y_te), y_tr.mean()))
    for n in comparison:
        comparison[n].update({f"test_{k}": v for k, v in test_results[n].items()})
    print("[TEST]", {n: {k: round(v, 3) for k, v in r.items()} for n, r in test_results.items()})
    print("[TEST] Baseline (rata-rata):", {k: round(v, 3) for k, v in baseline.items()})

    best_pipe = fitted[best_name]
    y_pred = best_pipe.predict(X_te)
    imp = grouped_importance(best_pipe) if hasattr(best_pipe.named_steps["model"], "feature_importances_") else {}

    make_eda_charts(df_raw, X, y, comparison, imp, y_te, y_pred, best_name)

    # --- Model akhir: di-fit ulang pada SELURUH data agar memakai semua informasi ---
    # (metrik yang dilaporkan tetap dari test set di atas)
    final_pipe = make_pipeline(candidate_models()[best_name]).fit(X, y)

    def choices(col, min_count=3):
        vc = X[col].value_counts()
        return sorted(vc[vc >= min_count].index.tolist())

    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "best_model": best_name,
        "target": TARGET, "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "n_total": len(X), "n_train": len(X_tr), "n_test": len(X_te),
        "comparison": comparison, "baseline_test": baseline,
        "test_metrics_best": test_results[best_name],
        "feature_importance": imp,
        "choices": {c: choices(c) for c in CATEGORICAL_FEATURES},
        "numeric_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC_FEATURES},
        "target_stats": {"mean": float(y.mean()), "std": float(y.std()),
                         "min": float(y.min()), "max": float(y.max()), "median": float(y.median())},
        "target_values": [float(v) for v in y.values],  # untuk grafik posisi skor di halaman hasil
        "n_countries": int(X["Country of Origin"].nunique()),
        "n_columns_raw": int(df_raw.shape[1]),
    }
    joblib.dump({"pipeline": final_pipe, "metadata": metadata}, MODEL_PATH)
    print("Model tersimpan:", MODEL_PATH)


if __name__ == "__main__":
    main()
