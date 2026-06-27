from django.db import models


class ModelMetrics(models.Model):
    name = models.CharField(max_length=80, default="RandomForestClassifier")
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1 = models.FloatField()
    confusion_matrix = models.JSONField()
    classes = models.JSONField()
    feature_importances = models.JSONField()
    n_samples = models.IntegerField()
    trained_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-trained_at"]
