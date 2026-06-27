from django.urls import path
from .views import KPIsView
urlpatterns = [path("kpis/", KPIsView.as_view(), name="dashboard-kpis")]
