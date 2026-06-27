"""Entrenamiento y predicción con Random Forest - versión optimizada para Render."""
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from apps.etl.models import Paciente
from .models import ModelMetrics

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "model.pkl"

FEATURES_ALL = [
    "edad", "imc", "glucosa", "colesterol",
    "presion_sistolica", "presion_diastolica",
    "frecuencia_cardiaca", "saturacion_oxigeno",
    "fumador", "consumo_alcohol", "antecedentes_familiares",
]
TARGET = "diagnostico_preliminar"

DIAG_MAP = {
    "hipertension":          "Hipertensión",
    "hipertensión":          "Hipertensión",
    "diabetes":              "Diabetes",
    "obesidad":              "Obesidad",
    "asma":                  "Asma",
    "cardiopatia":           "Cardiopatía",
    "cardiopatía":           "Cardiopatía",
    "epoc":                  "EPOC",
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


def _normalize_diag(value):
    if pd.isna(value) or str(value).strip() == "":
        return "Sin diagnóstico"
    s_clean = re.sub(r"[^a-z\s]", "", _strip_accents(str(value)).lower()).strip()
    for key, canonical in DIAG_MAP.items():
        if _strip_accents(key).lower() in s_clean:
            return canonical
    return str(value).strip().title()


def _get_available_features() -> list:
    paciente_fields = {f.name for f in Paciente._meta.get_fields()}
    return [f for f in FEATURES_ALL if f in paciente_fields]


def _dataset(features: list) -> pd.DataFrame:
    qs = Paciente.objects.exclude(
        diagnostico_preliminar=""
    ).exclude(
        diagnostico_preliminar__isnull=True
    ).values(*features, TARGET)

    df = pd.DataFrame(list(qs))
    if df.empty:
        raise RuntimeError("No hay pacientes con diagnóstico en la base de datos.")

    df[TARGET] = df[TARGET].apply(_normalize_diag)

    # Eliminar clases con menos de 2 ejemplos
    counts = df[TARGET].value_counts()
    df = df[df[TARGET].isin(counts[counts >= 2].index)]

    for c in ["fumador", "consumo_alcohol", "antecedentes_familiares"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    for c in features:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df


def train() -> ModelMetrics:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    features = _get_available_features()
    df = _dataset(features)

    if len(df) < 20:
        raise RuntimeError(f"Solo hay {len(df)} registros. Se necesitan al menos 20.")

    X = df[features].values
    y = df[TARGET].values

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    # Simple y liviano para Render
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,          # IMPORTANTE: n_jobs=1 en Render
        )),
    ])

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    classes = sorted(np.unique(y).tolist())

    metrics = ModelMetrics.objects.create(
        accuracy=float(accuracy_score(y_te, y_pred)),
        precision=float(precision_score(y_te, y_pred, average="weighted", zero_division=0)),
        recall=float(recall_score(y_te, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(y_te, y_pred, labels=classes).tolist(),
        classes=classes,
        feature_importances=dict(
            zip(features, [float(v) for v in clf.named_steps["model"].feature_importances_])
        ),
        n_samples=len(df),
    )

    joblib.dump({"model": clf, "features": features, "classes": classes}, MODEL_PATH)
    return metrics


def predict(paciente_id: int) -> dict:
    if not MODEL_PATH.exists():
        raise ValueError("Modelo no entrenado. Ejecuta train primero.")

    artifact = joblib.load(MODEL_PATH)
    clf      = artifact["model"]
    classes  = artifact["classes"]
    features = artifact["features"]

    try:
        paciente = Paciente.objects.get(pk=paciente_id)
    except Paciente.DoesNotExist:
        raise ValueError(f"Paciente {paciente_id} no encontrado.")

    row = {}
    for f in features:
        val = getattr(paciente, f, None)
        if val is None:
            raise ValueError(f"Campo '{f}' vacío en el paciente.")
        row[f] = int(val) if f in ("fumador", "consumo_alcohol", "antecedentes_familiares") else float(val)

    X = np.array([[row[f] for f in features]])
    proba = clf.predict_proba(X)[0]
    top_indices = np.argsort(proba)[::-1][:3]
    top3 = [
        {"diagnostico": classes[i], "confianza": round(float(proba[i]) * 100, 1)}
        for i in top_indices if proba[i] > 0
    ]

    return {
        "paciente_id": paciente_id,
        "nombre":      f"{paciente.nombres} {paciente.apellidos}",
        "diagnostico": top3[0]["diagnostico"],
        "confianza":   top3[0]["confianza"],
        "top3":        top3,
    }