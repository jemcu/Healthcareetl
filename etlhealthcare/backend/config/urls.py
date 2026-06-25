from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path("admin/", admin.site.urls),
    # HTML
    path("", include("apps.dashboard.urls")),
    # API
    path("api/auth/login/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/pacientes/", include("apps.etl.urls_pacientes")),
    path("api/etl/", include("apps.etl.urls")),
    path("api/analytics/", include("apps.analytics.urls")),
    path("api/dashboard/", include("apps.analytics.urls_dashboard")),
    path("api/predicciones/", include("apps.ml.urls")),
    path("api/reportes/", include("apps.reports.urls")),
    # Docs
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="docs"),
]
