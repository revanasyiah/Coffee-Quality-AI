"""
coffee_app/services.py
Jembatan antara Django dan model ML.
Model dimuat SEKALI (cache) dari file .joblib -> tidak ada training ulang saat aplikasi berjalan.
"""
import time
from functools import lru_cache

import joblib
import pandas as pd
from django.conf import settings

from ml.preprocessing import (
    DATASET_PATH,
    FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
    clean_features,
    load_dataset,
)


class GeminiAnalysisError(Exception):
    """Error yang aman ditampilkan saat analisis Gemini tidak dapat dibuat."""


@lru_cache(maxsize=1)
def get_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model belum ada di {MODEL_PATH}. Jalankan dulu: python ml/train_model.py")
    return joblib.load(MODEL_PATH)


def get_metadata():
    return get_bundle()["metadata"]


def categorize(score: float) -> dict:
    """Kategori berdasarkan ambang skala SCA (80 = specialty, 85 = excellent)."""
    if score >= 85:
        return {"label": "Excellent", "badge": "success",
                "text": "Estimasi masuk rentang Excellent (>= 85). Kopi dengan skor di rentang ini umumnya memiliki karakter rasa menonjol."}
    if score >= 80:
        return {"label": "Very Good", "badge": "warning",
                "text": "Estimasi masuk rentang Very Good (80 - 84,99), yaitu kategori specialty pada skala SCA."}
    return {"label": "Di bawah specialty", "badge": "secondary",
            "text": "Estimasi berada di bawah 80 poin, yaitu di bawah ambang specialty pada skala SCA."}


def predict_score(inputs: dict) -> float:
    """inputs: dict dengan kunci = nama kolom di ml.preprocessing.FEATURES."""
    bundle = get_bundle()
    raw = pd.DataFrame([{f: inputs.get(f) for f in FEATURES}])   # 1) DataFrame
    clean = clean_features(raw)                                  # 2) cleaning yang sama dgn training
    return float(bundle["pipeline"].predict(clean)[0])          # 3) pipeline: imputasi+encoding+model


def _generate_gemini_text(prompt: str) -> str:
    if not settings.GEMINI_API_KEY:
        raise GeminiAnalysisError(
            "API key Gemini belum diatur. Isi GEMINI_API_KEY pada file .env.")

    try:
        from google import genai
        from google.genai.errors import APIError
    except ImportError as exc:
        raise GeminiAnalysisError(
            "Paket Gemini belum terpasang. Jalankan: python -m pip install -r requirements.txt") from exc

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    models = [settings.GEMINI_MODEL]
    fallback_model = settings.GEMINI_FALLBACK_MODEL
    if fallback_model and fallback_model not in models:
        models.append(fallback_model)

    last_temporary_error = None
    for model_index, model in enumerate(models):
        attempts = 2 if model_index == 0 else 1
        for attempt in range(attempts):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
            except APIError as exc:
                if exc.code not in (429, 503):
                    raise GeminiAnalysisError(
                        f"Gemini gagal membuat analisis: {exc}") from exc
                last_temporary_error = exc
                if attempt + 1 < attempts:
                    time.sleep(1)
                continue

            analysis = (response.text or "").strip()
            if not analysis:
                raise GeminiAnalysisError(
                    "Gemini tidak mengembalikan analisis. Silakan coba lagi.")
            return analysis

    raise GeminiAnalysisError(
        "Model Gemini sedang padat. Percobaan ulang dan model cadangan juga gagal; "
        "silakan coba lagi beberapa saat.") from last_temporary_error


def generate_ai_analysis(prediction: dict) -> str:
    """Minta Gemini menjelaskan hasil prediksi dan karakteristik biji kopi."""
    prompt = (
        "Tulis analisis singkat dalam bahasa Indonesia untuk hasil estimasi kualitas kopi berikut. "
        "Jelaskan arti kategori dan skor secara hati-hati, hubungkan karakteristik input dengan "
        "kemungkinan hal yang perlu diperhatikan, lalu berikan 2-3 saran evaluasi atau penanganan. "
        "Jangan mengklaim bahwa fitur ini membuktikan rasa atau kualitas sensorik; skor berasal "
        "dari model prediksi dan bukan hasil cupping resmi. Jangan mengarang fakta yang tidak ada "
        "pada data.\n\n"
        f"Skor estimasi: {prediction['predicted_score']:.2f} dari 100\n"
        f"Kategori model: {prediction['category']}\n"
        f"Negara asal: {prediction['country']}\n"
        f"Varietas: {prediction['variety']}\n"
        f"Metode pengolahan: {prediction['processing_method']}\n"
        f"Warna biji: {prediction['color']}\n"
        f"Ketinggian: {prediction['altitude']} meter\n"
        f"Kadar air: {prediction['moisture']}%\n"
        f"Cacat kategori 1: {prediction['category_one_defects']}\n"
        f"Cacat kategori 2: {prediction['category_two_defects']}\n"
        f"Quakers: {prediction['quakers']}"
    )
    return _generate_gemini_text(prompt)


def generate_dashboard_chart_analysis(chart_type: str, metadata: dict) -> str:
    """Minta Gemini menginterpretasikan salah satu grafik distribusi pada dashboard."""
    values = metadata.get("target_values", [])
    if not values:
        raise GeminiAnalysisError("Data distribusi skor tidak tersedia untuk dianalisis.")

    if chart_type == "categories":
        counts = {
            "di bawah 80": sum(score < 80 for score in values),
            "80 sampai kurang dari 85 (Very Good)": sum(80 <= score < 85 for score in values),
            "85 ke atas (Excellent)": sum(score >= 85 for score in values),
        }
        details = (
            f"Jumlah sampel: {len(values)}\n"
            f"Jumlah skor di bawah 80: {counts['di bawah 80']}\n"
            f"Jumlah skor 80 sampai kurang dari 85: {counts['80 sampai kurang dari 85 (Very Good)']}\n"
            f"Jumlah skor 85 ke atas: {counts['85 ke atas (Excellent)']}"
        )
        chart_title = "Distribusi kategori kualitas"
    elif chart_type == "total_points":
        stats = metadata["target_stats"]
        details = (
            f"Jumlah sampel: {len(values)}\n"
            f"Rata-rata: {stats['mean']:.2f}\n"
            f"Median: {stats['median']:.2f}\n"
            f"Simpangan baku: {stats['std']:.2f}\n"
            f"Rentang skor: {stats['min']:.2f} sampai {stats['max']:.2f}\n"
            f"Jumlah skor 85 ke atas: {sum(score >= 85 for score in values)}"
        )
        chart_title = "Distribusi Total Cup Points"
    else:
        raise GeminiAnalysisError("Jenis grafik tidak dikenal.")

    prompt = (
        f"Analisis singkat grafik {chart_title} untuk aplikasi kualitas kopi dalam bahasa Indonesia. "
        "Jelaskan pola yang benar-benar didukung angka, apa artinya secara hati-hati, dan satu "
        "implikasi untuk membaca data. Jangan menganggap skor sebagai bukti rasa atau sertifikasi "
        "cupping; ini hanya distribusi skor dalam dataset. Jangan mengarang sebab atau fakta di luar data.\n\n"
        f"{details}"
    )
    return _generate_gemini_text(prompt)


def generate_insights_analysis(analysis_type: str, metadata: dict) -> str:
    """Minta Gemini menjelaskan satu bagian visualisasi dan evaluasi model."""
    comparison = metadata["comparison"]

    if analysis_type == "algorithm_comparison":
        rows = [
            f"{name}: CV MAE {values['cv_MAE']:.3f}, CV RMSE {values['cv_RMSE']:.3f}, "
            f"CV R² {values['cv_R2']:.3f}; uji MAE {values['test_MAE']:.3f}, "
            f"RMSE {values['test_RMSE']:.3f}, R² {values['test_R2']:.3f}"
            for name, values in comparison.items()
        ]
        baseline = metadata["baseline_test"]
        rows.append(
            f"Baseline rata-rata pada data uji: MAE {baseline['MAE']:.3f}, "
            f"RMSE {baseline['RMSE']:.3f}, R² {baseline['R2']:.3f}"
        )
        details = (
            f"Data latih untuk CV: {metadata['n_train']}; data uji terpisah: {metadata['n_test']}\n"
            f"Model terbaik menurut CV RMSE: {metadata['best_model']}\n" + "\n".join(rows)
        )
        title = "Perbandingan algoritma"
    elif analysis_type == "model_comparison":
        ordered = sorted(comparison.items(), key=lambda item: item[1]["cv_RMSE"])
        details = (
            f"RMSE CV 5-fold per model (lebih kecil lebih baik): "
            + "; ".join(f"{name}: {values['cv_RMSE']:.3f}" for name, values in ordered)
            + f"\nSimpangan baku skor sebagai baseline: {metadata['target_stats']['std']:.3f}"
            + f"\nModel terpilih: {metadata['best_model']}"
        )
        title = "Perbandingan model berdasarkan RMSE cross-validation"
    elif analysis_type == "actual_predicted":
        metrics = metadata["test_metrics_best"]
        details = (
            f"Model: {metadata['best_model']}\n"
            f"Jumlah sampel uji: {metadata['n_test']}\n"
            f"MAE uji: {metrics['MAE']:.3f}\n"
            f"RMSE uji: {metrics['RMSE']:.3f}\n"
            f"R² uji: {metrics['R2']:.3f}\n"
            "Grafik membandingkan skor aktual (sumbu X) dengan prediksi (sumbu Y); "
            "garis diagonal menunjukkan prediksi sempurna."
        )
        title = "Data uji aktual vs prediksi"
    elif analysis_type == "feature_importance":
        importance = metadata["feature_importance"]
        details = (
            f"Model: {metadata['best_model']}\n"
            + "\n".join(f"{name}: {value * 100:.2f}%" for name, value in importance.items())
        )
        title = "Feature importance Random Forest"
    elif analysis_type == "numeric_correlation":
        features, target_values = load_dataset()
        numeric = features[NUMERIC_FEATURES].copy()
        numeric[TARGET] = target_values.values
        correlations = numeric.corr()[TARGET].drop(TARGET)
        details = "\n".join(f"Korelasi Pearson {name} dengan {TARGET}: {value:.3f}"
                            for name, value in correlations.items())
        title = "Korelasi fitur numerik"
    elif analysis_type == "country_average":
        features, target_values = load_dataset()
        grouped = pd.DataFrame({
            "country": features["Country of Origin"],
            "score": target_values.values,
        }).groupby("country")["score"].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= 5].sort_values("mean")
        details = "\n".join(
            f"{country}: rata-rata {row['mean']:.2f}, n={int(row['count'])}"
            for country, row in grouped.iterrows()
        )
        title = "Rata-rata skor per negara (hanya negara dengan minimal 5 sampel)"
    elif analysis_type == "processing_method":
        features, target_values = load_dataset()
        grouped_data = pd.DataFrame({
            "method": features["Processing Method"],
            "score": target_values.values,
        })
        methods = ["Washed / Wet", "Natural / Dry", "Honey / Pulped Natural"]
        summaries = []
        for method in methods:
            scores = grouped_data.loc[grouped_data["method"] == method, "score"]
            if not scores.empty:
                summaries.append(
                    f"{method}: n={len(scores)}, min={scores.min():.2f}, "
                    f"Q1={scores.quantile(.25):.2f}, median={scores.median():.2f}, "
                    f"Q3={scores.quantile(.75):.2f}, max={scores.max():.2f}"
                )
        details = "\n".join(summaries)
        title = "Skor per metode pengolahan"
    elif analysis_type == "sensory_leakage":
        raw = pd.read_csv(DATASET_PATH)
        sensory = raw[LEAKAGE_COLUMNS + [TARGET]].loc[:, lambda data: data.std() > 0]
        correlations = sensory.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
        correlation_details = "\n".join(
            f"{name}: korelasi Pearson dengan {TARGET} = {value:.3f}"
            for name, value in correlations.items()
        )
        components = raw[LEAKAGE_COLUMNS].sum(axis=1, min_count=len(LEAKAGE_COLUMNS))
        max_difference = (components - raw[TARGET]).abs().max()
        details = (
            f"{TARGET} merupakan jumlah 10 skor sensorik. Selisih maksimum jumlah komponen "
            f"dengan target pada dataset: {max_difference:.3f}.\n{correlation_details}\n"
            "Fitur sensorik ini dikeluarkan dari input model agar model tidak membocorkan rumus target."
        )
        title = "Mengapa skor sensorik tidak dipakai (data leakage)"
    else:
        raise GeminiAnalysisError("Bagian visualisasi tidak dikenal.")

    prompt = (
        f"Jelaskan bagian '{title}' dalam bahasa Indonesia secara ringkas dan mudah dipahami. "
        "Analisis hanya berdasarkan angka berikut; jangan menyimpulkan sebab-akibat dari korelasi "
        "atau membuat klaim bahwa skor adalah penilaian cupping resmi. Sebutkan batas interpretasi "
        "data bila relevan.\n\n"
        f"{details}"
    )
    return _generate_gemini_text(prompt)
