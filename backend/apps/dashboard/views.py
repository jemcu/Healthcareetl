from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from apps.analytics import services as analytics
from apps.etl.models import Paciente, ETLRun
from apps.etl.engine import run_etl
from apps.ml.models import ModelMetrics
from apps.dashboard.forms import PacienteForm


def login_view(request):
    if request.method == "POST":
        u = authenticate(request,
                         username=request.POST.get("username"),
                         password=request.POST.get("password"))
        if u:
            login(request, u)
            return redirect("home")
        messages.error(request, "Credenciales inválidas")
    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required
def home(request):
    return render(request, "dashboard.html", {
        "kpis": analytics.kpis(),
        "riesgo": analytics.distribucion_riesgo(),
        "sexo": analytics.distribucion_sexo(),
        "imc": analytics.distribucion_imc(),
        "diagnosticos": analytics.top_diagnosticos(),
        "edad_seg": analytics.segmentacion_edad(),
        "criticos": analytics.pacientes_criticos(10),
    })


@login_required
def pacientes_view(request):
    qs = Paciente.objects.all()
    riesgo = request.GET.get("riesgo")
    if riesgo:
        qs = qs.filter(riesgo_enfermedad=riesgo)
    return render(request, "pacientes.html", {"pacientes": qs[:200], "riesgo": riesgo})


@login_required
def etl_view(request):
    if request.method == "POST" and request.FILES.get("file"):
        run = run_etl(request.FILES["file"], user=request.user)
        if run.status == "ok":
            messages.success(request, f"ETL ejecutado: {run.rows_loaded} pacientes cargados.")
        else:
            messages.error(request, f"ETL falló: {run.log[-200:]}")
        return redirect("etl")
    return render(request, "etl.html", {"runs": ETLRun.objects.all()[:30]})


@login_required
def ml_view(request):
    if request.method == "POST":
        try:
            from apps.ml.engine import train as train_model
            train_model()
            messages.success(request, "Modelo entrenado correctamente.")
        except Exception as exc:
            messages.error(request, str(exc))
        return redirect("ml")
    return render(request, "ml.html", {"metrics": ModelMetrics.objects.first()})


@login_required
def reportes_view(request):
    return render(request, "reportes.html", {"total": Paciente.objects.count()})


@login_required
def paciente_crear(request):
    if request.method == "POST":
        form = PacienteForm(request.POST)
        if form.is_valid():
            paciente = form.save()
            messages.success(
                request,
                f"Paciente {paciente.nombres} {paciente.apellidos} creado correctamente."
            )
            return redirect("pacientes")
        else:
            messages.error(request, "Corrige los errores del formulario.")
    else:
        form = PacienteForm()
    return render(request, "paciente_form.html", {"form": form})
