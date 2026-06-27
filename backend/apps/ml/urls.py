from django.urls import path
from .views import TrainView, MetricsView, PredictView

urlpatterns = [
    path("train/", TrainView.as_view(), name="ml-train"),
    path("metrics/", MetricsView.as_view(), name="ml-metrics"),
    path("predict/<int:paciente_id>/", PredictView.as_view(), name="ml-predict"),
]
