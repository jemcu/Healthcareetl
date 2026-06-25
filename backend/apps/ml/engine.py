"""Entrenamiento y predicción con Random Forest."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from apps.etl.models import Paciente
from .models import ModelMetrics

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "model.pkl"

FEATURES = ["edad", "imc", "glucosa", "colesterol", "presion_sistolica",
            "presion_diastolica", "frecuencia_cardiaca", "saturacion_oxigeno",
            "fumador", "consumo_alcohol", "antecedentes_familiares"]
TARGET = "riesgo_enfermedad"


def _dataset() -> pd.DataFrame:
    qs = Paciente.objects.all().values(*FEATURES, TARGET)
    df = pd.DataFrame(list(qs))
    for c in ["fumador", "consumo_alcohol", "antecedentes_familiares"]:
        df[c] = df[c].astype(int)
    return df


def train() -> ModelMetrics:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    from sklearn.model_selection import train_test_split

    df = _dataset()
    if len(df) < 50:
        raise RuntimeError("Se requieren al menos 50 pacientes para entrenar el modelo.")
    X = df[FEATURES].values
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
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
        feature_importances=dict(zip(FEATURES, [float(v) for v in clf.feature_importances_])),
        n_samples=len(df),
    )
    joblib.dump({"model": clf, "features": FEATURES, "classes": classes}, MODEL_PATH)
    return metrics


def predict(payload: dict) -> dict:
    from sklearn.ensemble import RandomForestClassifier  # necesario para deserializar joblib

    if not MODEL_PATH.exists():
        raise RuntimeError("Modelo no entrenado. Ejecuta /api/ml/train/ o `manage.py train_model`.")
    bundle = joblib.load(MODEL_PATH)
    clf, feats, classes = bundle["model"], bundle["features"], bundle["classes"]
    row = [float(payload.get(f, 0) or 0) for f in feats]
    proba = clf.predict_proba([row])[0]
    pred_idx = int(np.argmax(proba))
    return {
        "riesgo_predicho": classes[pred_idx],
        "probabilidades": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }
