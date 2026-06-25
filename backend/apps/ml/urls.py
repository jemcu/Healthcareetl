from django.urls import path
from .views import TrainView, MetricsView, PredictView

urlpatterns = [
    path("", PredictView.as_view(), name="predict"),
    path("train/", TrainView.as_view(), name="train"),
    path("metrics/", MetricsView.as_view(), name="metrics"),
]
