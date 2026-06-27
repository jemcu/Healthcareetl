"""Normaliza los valores corruptos de diagnostico_preliminar en BD."""
import re
import unicodedata
from django.core.management.base import BaseCommand
from apps.etl.models import Paciente

DIAG_MAP = {
    "hipertension":          "Hipertensión",
    "hipertensión":          "Hipertensión",
    "hipertensi":            "Hipertensión",
    "hta":                   "Hipertensión",
    "diabetes":              "Diabetes",
    "diabete":               "Diabetes",
    "obesidad":              "Obesidad",
    "obeso":                 "Obesidad",
    "cardiopatia":           "Cardiopatía",
    "cardiopatía":           "Cardiopatía",
    "epoc":                  "EPOC",
    "asma":                  "Asma",
    "anemia":                "Anemia",
    "riesgo cardiovascular": "Riesgo Cardiovascular",
    "cardiovascular":        "Riesgo Cardiovascular",
    "normal":                "Sin diagnóstico",
    "sano":                  "Sin diagnóstico",
    "ninguno":               "Sin diagnóstico",
    "sin diagnostico":       "Sin diagnóstico",
    "sin diagnóstico":       "Sin diagnóstico",
    "ninguna":               "Sin diagnóstico",
    "no aplica":             "Sin diagnóstico",
}

def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def _normalize(value: str) -> str:
    if not value or not value.strip():
        return "Sin diagnóstico"
    s_clean = re.sub(r"[^a-z\s]", "", _strip_accents(value).lower()).strip()
    for key, canonical in DIAG_MAP.items():
        if _strip_accents(key).lower() in s_clean:
            return canonical
    return value.strip().title()


class Command(BaseCommand):
    help = "Corrige encoding de diagnostico_preliminar en todos los pacientes"

    def handle(self, *args, **options):
        pacientes = Paciente.objects.exclude(
            diagnostico_preliminar__isnull=True
        ).exclude(diagnostico_preliminar="")

        total = pacientes.count()
        corregidos = 0

        for p in pacientes:
            nuevo = _normalize(p.diagnostico_preliminar)
            if nuevo != p.diagnostico_preliminar:
                p.diagnostico_preliminar = nuevo
                p.save(update_fields=["diagnostico_preliminar"])
                corregidos += 1

        self.stdout.write(self.style.SUCCESS(
            f"✓ Corregidos {corregidos} de {total} pacientes."
        ))
