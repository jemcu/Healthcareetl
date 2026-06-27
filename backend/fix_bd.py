import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

import re, unicodedata
from apps.etl.models import Paciente

DIAG_MAP = {
    "hipertension": "Hipertensión",
    "diabetes": "Diabetes",
    "obesidad": "Obesidad",
    "cardiopatia": "Cardiopatía",
    "epoc": "EPOC",
    "asma": "Asma",
    "anemia": "Anemia",
    "cardiovascular": "Riesgo Cardiovascular",
    "normal": "Sin diagnóstico",
    "sano": "Sin diagnóstico",
    "ninguno": "Sin diagnóstico",
    "sin diag": "Sin diagnóstico",
}

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize(value):
    if not value: return "Sin diagnóstico"
    s = re.sub(r"[^a-z\s]", "", strip_accents(str(value)).lower()).strip()
    for key, canonical in DIAG_MAP.items():
        if strip_accents(key).lower() in s:
            return canonical
    return value.strip().title()

pacientes = Paciente.objects.exclude(diagnostico_preliminar__isnull=True).exclude(diagnostico_preliminar="")
corregidos = 0
for p in pacientes:
    nuevo = normalize(p.diagnostico_preliminar)
    if nuevo != p.diagnostico_preliminar:
        p.diagnostico_preliminar = nuevo
        p.save(update_fields=["diagnostico_preliminar"])
        corregidos += 1

print(f"Corregidos: {corregidos} de {pacientes.count()} pacientes")

# Mostrar clases únicas resultantes
from django.db.models import Count
clases = Paciente.objects.values("diagnostico_preliminar").annotate(n=Count("id_paciente")).order_by("-n")
for c in clases:
    print(f"  {c['diagnostico_preliminar']}: {c['n']}")