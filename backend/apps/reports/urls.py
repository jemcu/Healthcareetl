from django.urls import path
from .views import ReportView

urlpatterns = [path("<str:tipo>/", ReportView.as_view(), name="report")]
