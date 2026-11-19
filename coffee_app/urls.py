from django.urls import path
from . import views

app_name = "coffee_app"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("prediksi/", views.predict, name="predict"),
    path("hasil/<int:pk>/", views.result, name="result"),
    path("hasil/<int:pk>/analisis-ai/", views.analyze_result, name="analyze_result"),
    path("analisis-grafik/<slug:chart_type>/", views.analyze_dashboard_chart, name="analyze_dashboard_chart"),
    path("visualisasi/analisis-ai/<slug:analysis_type>/", views.analyze_insight, name="analyze_insight"),
    path("visualisasi/", views.insights, name="insights"),
    path("riwayat/", views.history, name="history"),
]
