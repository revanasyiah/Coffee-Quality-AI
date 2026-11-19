import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.contrib import messages
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import PredictionForm
from .models import PredictionHistory
from .services import (
    GeminiAnalysisError,
    categorize,
    generate_ai_analysis,
    generate_dashboard_chart_analysis,
    generate_insights_analysis,
    get_metadata,
    predict_score,
)


def _position_chart(score, values, rmse):
    """Histogram data latih + garis estimasi (base64 PNG)."""
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(values, bins=20, color="#d9bfa6", edgecolor="white", label="Data kopi pada dataset")
    ax.axvspan(score - rmse, score + rmse, color="#b08968", alpha=.25, label=f"Estimasi ± RMSE ({rmse:.2f})")
    ax.axvline(score, color="#4b2e1e", lw=2.5, label=f"Estimasi Anda: {score:.2f}")
    ax.set_xlabel("Total Cup Points"); ax.set_ylabel("Jumlah sampel")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=120); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _render_dashboard(request, chart_analysis=None, chart_error=None, chart_type=None):
    return render(request, "coffee_app/dashboard.html", {
        "meta": get_metadata(),
        "page": "dashboard",
        "dashboard_analysis": chart_analysis,
        "dashboard_analysis_error": chart_error,
        "dashboard_analysis_type": chart_type,
    })


def dashboard(request):
    return _render_dashboard(request)


@require_POST
def analyze_dashboard_chart(request, chart_type):
    if chart_type not in {"categories", "total_points"}:
        raise Http404
    try:
        analysis = generate_dashboard_chart_analysis(chart_type, get_metadata())
    except GeminiAnalysisError as exc:
        return _render_dashboard(request, chart_error=str(exc), chart_type=chart_type)
    return _render_dashboard(request, chart_analysis=analysis, chart_type=chart_type)


def predict(request):
    form = PredictionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        score = predict_score(form.to_model_inputs())
        score = max(0.0, min(100.0, score))  # batas skala cupping
        for w in form.out_of_range_warnings():
            messages.warning(request, w)
        d = form.cleaned_data
        rec = PredictionHistory.objects.create(
            altitude=d["altitude"], moisture=d["moisture"],
            category_one_defects=d["category_one_defects"], category_two_defects=d["category_two_defects"],
            quakers=d["quakers"], country=d["country"], variety=d["variety"],
            processing_method=d["processing_method"], color=d["color"],
            predicted_score=round(score, 2), category=categorize(score)["label"])
        return redirect("coffee_app:result", pk=rec.pk)  # POST -> redirect -> GET
    return render(request, "coffee_app/predict.html", {"form": form, "page": "predict"})


def _render_result(request, rec, ai_analysis=None, ai_error=None):
    meta = get_metadata()
    rmse = meta["test_metrics_best"]["RMSE"]
    ctx = {
        "rec": rec, "meta": meta, "cat": categorize(rec.predicted_score), "rmse": rmse,
        "low": rec.predicted_score - rmse, "high": rec.predicted_score + rmse,
        "chart": _position_chart(rec.predicted_score, meta["target_values"], rmse),
        "ai_analysis": ai_analysis,
        "ai_error": ai_error,
        "page": "predict",
    }
    return render(request, "coffee_app/result.html", ctx)


def result(request, pk):
    rec = get_object_or_404(PredictionHistory, pk=pk)
    return _render_result(request, rec)


@require_POST
def analyze_result(request, pk):
    rec = get_object_or_404(PredictionHistory, pk=pk)
    try:
        analysis = generate_ai_analysis({
            "predicted_score": rec.predicted_score,
            "category": rec.category,
            "country": rec.country,
            "variety": rec.variety,
            "processing_method": rec.processing_method,
            "color": rec.color,
            "altitude": rec.altitude,
            "moisture": rec.moisture,
            "category_one_defects": rec.category_one_defects,
            "category_two_defects": rec.category_two_defects,
            "quakers": rec.quakers,
        })
    except GeminiAnalysisError as exc:
        return _render_result(request, rec, ai_error=str(exc))
    return _render_result(request, rec, ai_analysis=analysis)


def _render_insights(request, analysis=None, error=None, analysis_type=None):
    meta = get_metadata()
    rows = [{"name": n, **v} for n, v in meta["comparison"].items()]
    imp = [(k, round(v * 100, 1)) for k, v in meta["feature_importance"].items()]
    return render(request, "coffee_app/insights.html", {
        "meta": meta,
        "rows": rows,
        "imp": imp,
        "page": "insights",
        "insights_analysis": analysis,
        "insights_analysis_error": error,
        "insights_analysis_type": analysis_type,
    })


def insights(request):
    return _render_insights(request)


@require_POST
def analyze_insight(request, analysis_type):
    analysis_types = {
        "algorithm_comparison", "model_comparison", "actual_predicted",
        "feature_importance", "numeric_correlation", "country_average",
        "processing_method", "sensory_leakage",
    }
    if analysis_type not in analysis_types:
        raise Http404
    try:
        analysis = generate_insights_analysis(analysis_type, get_metadata())
    except GeminiAnalysisError as exc:
        return _render_insights(request, error=str(exc), analysis_type=analysis_type)
    return _render_insights(request, analysis=analysis, analysis_type=analysis_type)


def history(request):
    items = PredictionHistory.objects.all()[:50]
    return render(request, "coffee_app/history.html", {"items": items, "page": "history"})
