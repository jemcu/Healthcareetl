"""Entrenamiento y predicción con HistGradientBoosting - versión optimizada y liviana para Render."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from apps.etl.diagnosis import normalize as _normalize_diag
from apps.etl.models import Paciente
from .models import ModelMetrics

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "model.pkl"

FEATURES_BASE = [
    "edad", "imc", "glucosa", "colesterol",
    "presion_sistolica", "presion_diastolica",
    "frecuencia_cardiaca", "saturacion_oxigeno",
    "fumador", "consumo_alcohol", "antecedentes_familiares",
]

FEATURES_ENGINEERED = [
    "imc_glucosa",
    "edad_presion_sistolica",
]

TARGET = "diagnostico_preliminar"


def _get_available_features() -> list:
    paciente_fields = {f.name for f in Paciente._meta.get_fields()}
    return [f for f in FEATURES_BASE if f in paciente_fields]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["imc_glucosa"] = df["imc"] * df["glucosa"]
    df["edad_presion_sistolica"] = df["edad"] * df["presion_sistolica"]
    return df


def _engineer_features_dict(row: dict) -> dict:
    row = dict(row)
    row["imc_glucosa"] = row["imc"] * row["glucosa"]
    row["edad_presion_sistolica"] = row["edad"] * row["presion_sistolica"]
    return row


def _dataset(base_features: list) -> pd.DataFrame:
    qs = Paciente.objects.exclude(
        diagnostico_preliminar=""
    ).exclude(
        diagnostico_preliminar__isnull=True
    ).values(*base_features, TARGET)

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

    for c in base_features:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Ingenieria de caracteristicas (interacciones)
    df = _engineer_features(df)

    all_feats = base_features + FEATURES_ENGINEERED
    for c in all_feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def train() -> ModelMetrics:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    base_features = _get_available_features()
    df = _dataset(base_features)

    if len(df) < 20:
        raise RuntimeError(f"Solo hay {len(df)} registros. Se necesitan al menos 20.")

    all_features = base_features + FEATURES_ENGINEERED

    non_num = [c for c in all_features if df[c].dtype == "object"]
    if non_num:
        raise TypeError(
            f"Columnas no numéricas detectadas: {non_num}. "
            "Revisa que los datos cargados sean válidos."
        )

    X = df[all_features].values
    y = df[TARGET].values

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # HistGradientBoostingClassifier:
    #   - Maneja valores nulos nativamente
    #   - No necesita scaling (es tree-based)
    #   - Secuencial (sin n_jobs) ideal para Render con 512 MB
    #   - Generalmente mas preciso que RF en datos tabulares
    #   - class_weight="balanced" ayuda con clases desbalanceadas
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=20,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_bins=64,
        l2_regularization=0.1,
        random_state=42,
    )

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
            zip(all_features, [float(v) for v in clf.feature_importances_])
        ),
        n_samples=len(df),
    )

    joblib.dump({"model": clf, "features": all_features, "classes": classes}, MODEL_PATH)
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

    # Construir dict con features base desde el objeto Paciente
    row = {}
    for f in FEATURES_BASE:
        val = getattr(paciente, f, None)
        if val is None:
            raise ValueError(f"Campo '{f}' vacio en el paciente.")
        row[f] = int(val) if f in ("fumador", "consumo_alcohol", "antecedentes_familiares") else float(val)

    # Ingenieria de caracteristicas
    row = _engineer_features_dict(row)

    # Construir vector en el orden exacto que uso el modelo entrenado
    try:
        X = np.array([[float(row[f]) for f in features]], dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise TypeError(f"Error al convertir features a numérico para predicción: {exc}")
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