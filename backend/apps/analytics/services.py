"""Analítica: KPIs y estadística descriptiva sobre pacientes."""
from django.db.models import Avg, Count, Q
from apps.etl.models import Paciente


def kpis() -> dict:
    qs = Paciente.objects.all()
    total = qs.count()
    if not total:
        return {"total": 0}

    return {
        "total": total,
        "criticos": qs.filter(riesgo_enfermedad="critico").count(),
        "alto_riesgo": qs.filter(riesgo_enfermedad="alto").count(),
        "hipertensos": qs.filter(presion_sistolica__gte=140).count(),
        "diabeticos": qs.filter(glucosa__gte=126).count(),
        "fumadores": qs.filter(fumador=True).count(),
        "obesidad": qs.filter(imc__gte=30).count(),
        "imc_promedio": round(qs.aggregate(v=Avg("imc"))["v"] or 0, 2),
        "edad_promedio": round(qs.aggregate(v=Avg("edad"))["v"] or 0, 1),
        "glucosa_promedio": round(qs.aggregate(v=Avg("glucosa"))["v"] or 0, 1),
        "presion_promedio": round(qs.aggregate(v=Avg("presion_sistolica"))["v"] or 0, 1),
    }


def distribucion_riesgo():
    return list(
        Paciente.objects.values("riesgo_enfermedad")
        .annotate(total=Count("id")).order_by("riesgo_enfermedad")
    )


def distribucion_sexo():
    return list(Paciente.objects.values("sexo").annotate(total=Count("id")))


def distribucion_imc():
    return list(
        Paciente.objects.values("imc_clasificacion")
        .annotate(total=Count("id")).order_by("imc_clasificacion")
    )


def top_diagnosticos(limit=8):
    return list(
        Paciente.objects.values("diagnostico_preliminar")
        .annotate(total=Count("id")).order_by("-total")[:limit]
    )


def segmentacion_edad():
    rangos = [(0, 17, "0-17"), (18, 29, "18-29"), (30, 44, "30-44"),
              (45, 59, "45-59"), (60, 74, "60-74"), (75, 120, "75+")]
    out = []
    for lo, hi, label in rangos:
        qs = Paciente.objects.filter(edad__gte=lo, edad__lte=hi)
        out.append({
            "rango": label,
            "total": qs.count(),
            "criticos": qs.filter(riesgo_enfermedad="critico").count(),
            "alto": qs.filter(riesgo_enfermedad="alto").count(),
        })
    return out


def estadistica_descriptiva():
    import statistics
    fields = ["edad", "imc", "glucosa", "colesterol", "presion_sistolica",
              "presion_diastolica", "frecuencia_cardiaca", "saturacion_oxigeno"]
    out = {}
    for f in fields:
        vals = list(Paciente.objects.values_list(f, flat=True))
        if not vals:
            continue
        try:
            moda = statistics.mode(vals)
        except statistics.StatisticsError:
            moda = vals[0]
        out[f] = {
            "media": round(statistics.fmean(vals), 2),
            "mediana": round(statistics.median(vals), 2),
            "moda": round(moda, 2) if isinstance(moda, (int, float)) else moda,
            "desv_std": round(statistics.pstdev(vals), 2),
            "min": min(vals), "max": max(vals),
        }
    return out


def pacientes_criticos(limit=20):
    qs = Paciente.objects.filter(
        Q(presion_sistolica__gt=180) | Q(glucosa__gt=300) | Q(saturacion_oxigeno__lt=85)
        | Q(riesgo_enfermedad="critico")
    )[:limit]
    return [
        {"id": p.id_paciente, "nombre": f"{p.nombres} {p.apellidos}",
         "edad": p.edad, "presion": p.presion_sistolica, "glucosa": p.glucosa,
         "spo2": p.saturacion_oxigeno, "riesgo": p.riesgo_enfermedad}
        for p in qs
    ]
