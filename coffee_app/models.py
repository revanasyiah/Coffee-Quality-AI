from django.db import models


class PredictionHistory(models.Model):
    """Riwayat prediksi (SQLite via Django ORM)."""
    created_at = models.DateTimeField(auto_now_add=True)
    altitude = models.FloatField()
    moisture = models.FloatField()
    category_one_defects = models.PositiveIntegerField()
    category_two_defects = models.PositiveIntegerField()
    quakers = models.PositiveIntegerField()
    country = models.CharField(max_length=100)
    variety = models.CharField(max_length=100)
    processing_method = models.CharField(max_length=100)
    color = models.CharField(max_length=50)
    predicted_score = models.FloatField()
    category = models.CharField(max_length=40)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.country} - {self.predicted_score:.2f}"
