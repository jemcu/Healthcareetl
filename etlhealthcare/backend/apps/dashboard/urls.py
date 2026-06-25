from django.urls import path
from . import views

urlpatterns = [
    path("",                   views.home,                  name="home"),
    path("login/",             views.login_view,            name="login"),
    path("logout/",            views.logout_view,           name="logout"),
    path("pacientes/",         views.pacientes_view,        name="pacientes"),
    path("pacientes/crear/",   views.paciente_crear,        name="paciente_crear"),
    path("pacientes/<int:paciente_id>/predict/", views.paciente_predict_ajax, name="paciente_predict"),
    path("etl/",               views.etl_view,              name="etl"),
    path("ml/",                views.ml_view,               name="ml"),
    path("reportes/",          views.reportes_view,         name="reportes"),
]
