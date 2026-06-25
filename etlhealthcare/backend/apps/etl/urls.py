from django.urls import path
from .views import ETLRunView, ETLHistoryView

urlpatterns = [
    path("run/", ETLRunView.as_view(), name="etl-run"),
    path("history/", ETLHistoryView.as_view(), name="etl-history"),
]
