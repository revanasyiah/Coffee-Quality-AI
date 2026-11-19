from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ("created_at", "country", "variety", "processing_method", "predicted_score", "category")
    list_filter = ("category", "country")
