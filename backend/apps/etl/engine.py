"""Motor ETL: extract, transform, load para datos clínicos."""
from __future__ import annotations

import io
import re
import time
import unicodedata
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from django.db import transaction

from .models import ETLRun, Paciente

from apps.etl.diagnosis import normalize as _diagnosis_normalize


def _normalize_diag(value):
    """Wrapper que usa fallback_title=True para datos crudos del Excel."""
    return _diagnosis_normalize(value, fallback_title=True)


NUM_WORDS_ES = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11,
    "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "veinte": 20,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
    "setenta": 70, "ochenta": 80, "noventa": 90, "cien": 100,
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _to_int(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value) if not np.isnan(value) else np.nan
    s = _strip_accents(str(value)).strip().lower()
    if s in NUM_WORDS_ES:
        return NUM_WORDS_ES[s]
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else np.nan


def _to_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).replace(",", ".").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def _to_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    s = _strip_accents(str(value)).strip().lower()
    return s in {"true", "1", "si", "yes", "y", "verdadero", "v"}



def _normalize_sexo(value):
    if pd.isna(value):
        return "O"
    s = _strip_accents(str(value)).strip().lower()
    if s.startswith("m") or s in {"masculino", "hombre", "h"}:
        return "M"
    if s.startswith("f") or s in {"femenino", "mujer"}:
        return "F"
    return "O"


def _clip(s: pd.Series, lo: float, hi: float, fill_strategy: str = "median") -> Tuple[pd.Series, int]:
    mask = (s < lo) | (s > hi)
    n_out = int(mask.sum())
    if n_out:
        fill = s.median() if fill_strategy == "median" else s.mean()
        s = s.mask(mask, fill)
    return s, n_out


def _classify_imc(imc: float) -> str:
    if imc < 18.5: return "bajo_peso"
    if imc < 25:   return "normal"
    if imc < 30:   return "sobrepeso"
    return "obesidad"


def _classify_risk(row) -> str:
    score = 0
    if row["presion_sistolica"] > 180 or row["glucosa"] > 300 or row["saturacion_oxigeno"] < 85:
        return "critico"
    if row["presion_sistolica"] >= 140: score += 2
    elif row["presion_sistolica"] >= 130: score += 1
    if row["glucosa"] >= 200: score += 2
    elif row["glucosa"] >= 126: score += 1
    if row["colesterol"] >= 240: score += 2
    elif row["colesterol"] >= 200: score += 1
    if row["imc"] >= 30: score += 2
    elif row["imc"] >= 25: score += 1
    if row["fumador"]: score += 1
    if row["edad"] >= 60: score += 1
    if score >= 6: return "alto"
    if score >= 3: return "medio"
    return "bajo"


COLUMN_ALIASES = {
    "presion_sistolica": ["presion_sistolica", "presión_sistólica", "presion sistolica"],
    "presion_diastolica": ["presion_diastolica", "presión_diastólica"],
    "frecuencia_cardiaca": ["frecuencia_cardiaca", "frecuencia cardiaca", "fc"],
    "saturacion_oxigeno": ["saturacion_oxigeno", "saturación_oxígeno", "spo2"],
    "antecedentes_familiares": ["antecedentes_familiares", "antecedentes"],
    "actividad_fisica": ["actividad_fisica", "actividad_física"],
    "diagnostico_preliminar": ["diagnostico_preliminar", "diagnóstico_preliminar"],
    "riesgo_enfermedad": ["riesgo_enfermedad", "riesgo"],
    "fecha_consulta": ["fecha_consulta", "fecha"],
}


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c: _strip_accents(c).lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=lower)
    for canonical, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            a2 = _strip_accents(a).lower().replace(" ", "_")
            if a2 in df.columns and canonical not in df.columns:
                df = df.rename(columns={a2: canonical})
    return df


def extract(path_or_buffer) -> Tuple[pd.DataFrame, str]:
    source = str(getattr(path_or_buffer, "name", path_or_buffer))
    ext = source.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(path_or_buffer)
    else:
        df = pd.read_csv(path_or_buffer)
    return df, source


def transform(df: pd.DataFrame, log: list[str]) -> Tuple[pd.DataFrame, dict]:
    metrics = {"duplicates_removed": 0, "nulls_imputed": 0, "outliers_fixed": 0}

    df = _canonicalize_columns(df)

    n0 = len(df)
    df = df.drop_duplicates(subset=["id_paciente"], keep="first") if "id_paciente" in df.columns else df.drop_duplicates()
    metrics["duplicates_removed"] = n0 - len(df)
    log.append(f"Duplicados eliminados: {metrics['duplicates_removed']}")

    int_cols = ["id_paciente", "edad", "presion_sistolica", "presion_diastolica", "frecuencia_cardiaca"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].map(_to_int)

    float_cols = ["peso", "altura", "imc", "glucosa", "colesterol", "saturacion_oxigeno", "temperatura"]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(_to_float)

    for c in ["antecedentes_familiares", "fumador", "consumo_alcohol"]:
        if c in df.columns:
            df[c] = df[c].map(_to_bool)

    if "sexo" in df.columns:
        df["sexo"] = df["sexo"].map(_normalize_sexo)
    if "diagnostico_preliminar" in df.columns:
        df["diagnostico_preliminar"] = df["diagnostico_preliminar"].map(_normalize_diag)
    if "actividad_fisica" in df.columns:
        df["actividad_fisica"] = df["actividad_fisica"].fillna("desconocida").astype(str).str.lower().str.strip()

    if "fecha_consulta" in df.columns:
        df["fecha_consulta"] = pd.to_datetime(df["fecha_consulta"], errors="coerce").dt.date

    null_before = int(df[int_cols + float_cols].isna().sum().sum())
    for c, lo, hi in [("edad", 0, 110), ("presion_sistolica", 70, 220),
                      ("presion_diastolica", 40, 140), ("frecuencia_cardiaca", 35, 200)]:
        if c in df.columns:
            df[c], n = _clip(df[c], lo, hi)
            metrics["outliers_fixed"] += n
            df[c] = df[c].fillna(df[c].median()).astype(int)

    for c, lo, hi in [("peso", 25, 250), ("altura", 1.2, 2.2),
                      ("glucosa", 40, 500), ("colesterol", 80, 400),
                      ("saturacion_oxigeno", 60, 100), ("temperatura", 33, 42)]:
        if c in df.columns:
            df[c], n = _clip(df[c], lo, hi)
            metrics["outliers_fixed"] += n
            df[c] = df[c].fillna(df[c].median())

    null_after = int(df[int_cols + float_cols].isna().sum().sum())
    metrics["nulls_imputed"] = max(null_before - null_after, 0)
    log.append(f"Outliers corregidos: {metrics['outliers_fixed']}")
    log.append(f"Nulos imputados: {metrics['nulls_imputed']}")

    df["imc"] = (df["peso"] / (df["altura"] ** 2)).round(2)
    df["imc_clasificacion"] = df["imc"].map(_classify_imc)
    df["riesgo_enfermedad"] = df.apply(_classify_risk, axis=1)

    for c in ["nombres", "apellidos"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip().str.title()

    return df, metrics


@transaction.atomic
def load(df: pd.DataFrame) -> int:
    Paciente.objects.all().delete()
    objs = []
    for _, r in df.iterrows():
        objs.append(Paciente(
            id_paciente=int(r["id_paciente"]),
            nombres=r.get("nombres", "") or "",
            apellidos=r.get("apellidos", "") or "",
            edad=int(r["edad"]),
            sexo=r["sexo"],
            peso=float(r["peso"]),
            altura=float(r["altura"]),
            imc=float(r["imc"]),
            imc_clasificacion=r["imc_clasificacion"],
            presion_sistolica=int(r["presion_sistolica"]),
            presion_diastolica=int(r["presion_diastolica"]),
            frecuencia_cardiaca=int(r["frecuencia_cardiaca"]),
            glucosa=float(r["glucosa"]),
            colesterol=float(r["colesterol"]),
            saturacion_oxigeno=float(r["saturacion_oxigeno"]),
            temperatura=float(r["temperatura"]),
            antecedentes_familiares=bool(r.get("antecedentes_familiares", False)),
            fumador=bool(r.get("fumador", False)),
            consumo_alcohol=bool(r.get("consumo_alcohol", False)),
            actividad_fisica=str(r.get("actividad_fisica", "") or "")[:30],
            diagnostico_preliminar=str(r.get("diagnostico_preliminar", "") or "")[:120],
            riesgo_enfermedad=r["riesgo_enfermedad"],
            fecha_consulta=r.get("fecha_consulta"),
        ))
    Paciente.objects.bulk_create(objs, batch_size=500)
    return len(objs)


def run_etl(path_or_buffer, user=None) -> ETLRun:
    t0 = time.perf_counter()
    log: list[str] = []
    try:
        df, source = extract(path_or_buffer)
        log.append(f"Extracción: {len(df)} filas desde {source}")
        rows_extracted = len(df)
        df, metrics = transform(df, log)
        rows_after = len(df)
        rows_loaded = load(df)
        log.append(f"Carga: {rows_loaded} pacientes insertados")
        run = ETLRun.objects.create(
            source=source, user=user,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            rows_extracted=rows_extracted,
            rows_after_transform=rows_after,
            rows_loaded=rows_loaded,
            duplicates_removed=metrics["duplicates_removed"],
            nulls_imputed=metrics["nulls_imputed"],
            outliers_fixed=metrics["outliers_fixed"],
            status="ok",
            log="\n".join(log),
        )
        return run
    except Exception as exc:  # noqa: BLE001
        log.append(f"ERROR: {exc}")
        return ETLRun.objects.create(
            source=str(getattr(path_or_buffer, "name", path_or_buffer)),
            user=user,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            status="error",
            log="\n".join(log),
        )
