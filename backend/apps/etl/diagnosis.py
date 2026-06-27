"""
Módulo compartido de normalización de diagnósticos.
Usado por el motor ETL, el comando fix_diagnostico y el motor ML.
Centraliza DIAG_MAP y la lógica de normalización para evitar duplicación.
"""
from __future__ import annotations

import re
import unicodedata

# ── Mapeo de variantes → diagnóstico canónico ──────────────────────────────
DIAG_MAP: dict[str, str] = {
    "hipertension":          "Hipertensión",
    "hipertensión":          "Hipertensión",
    "hipertensi":            "Hipertensión",
    "hipertencion":          "Hipertensión",
    "hipertención":          "Hipertensión",
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

# Valores canónicos (para verificación rápida)
CANONICAL_VALUES: set[str] = set(DIAG_MAP.values())

# Detección opcional de pandas (sin dependencia forzada)
try:
    import pandas as pd

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ── Utilidades ──────────────────────────────────────────────────────────────
def strip_accents(s: str) -> str:
    """Elimina tildes / diacríticos de una cadena Unicode."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize(value, fallback_title: bool = False) -> str:
    """
    Normaliza un valor de diagnóstico a su forma canónica.

    Parámetros
    ----------
    value:
        Valor a normalizar (str, None, pd.NA, etc.).
    fallback_title:
        Si *True*, aplica .title() a valores no reconocidos
        (útil durante ETL con datos crudos del Excel).
        Si *False*, devuelve el valor original intacto (idempotente).

    Returns
    -------
    str
        Valor canónico del diagnóstico (o el original si no se reconoce).
    """
    # ── Nulos / vacíos ──────────────────────────────────────────────────
    if value is None:
        return "Sin diagnóstico"

    if _HAS_PANDAS:
        try:
            if pd.isna(value):
                return "Sin diagnóstico"
        except (TypeError, ValueError):
            pass

    s = str(value)
    if not s.strip():
        return "Sin diagnóstico"

    # ── Limpiar y buscar en el mapa ─────────────────────────────────────
    s_clean = re.sub(r"[^a-z\s]", "", strip_accents(s).lower()).strip()
    if not s_clean:
        return "Sin diagnóstico"

    for key, canonical in DIAG_MAP.items():
        if strip_accents(key).lower() in s_clean:
            return canonical

    # ── Sin coincidencia ────────────────────────────────────────────────
    if fallback_title:
        return s.strip().title()
    return s.strip()
