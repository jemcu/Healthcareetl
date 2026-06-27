"""Entrenamiento y predicción con RandomForest optimizado,
feature engineering avanzado y selección de características."""
from __future__ import annotations

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

# ── Features base del modelo Paciente ──────────────────────────
FEATURES_NUMERIC = [
    "edad", "imc", "glucosa", "colesterol",
    "presion_sistolica", "presion_diastolica",
    "frecuencia_cardiaca", "saturacion_oxigeno",
    "temperatura", "peso", "altura",
]
FEATURES_BOOL = ["fumador", "consumo_alcohol", "antecedentes_familiares"]
FEATURES_CATEGORICAL = ["sexo", "actividad_fisica"]
# Campos derivados del ETL que también sirven como features
FEATURES_DERIVED = ["imc_clasificacion", "riesgo_enfermedad"]

TARGET = "diagnostico_preliminar"

# ── Normalización de diagnósticos ─────────────────────────────
DIAG_MAP = {
    "hipertension": "Hipertensión", "hipertensión": "Hipertensión",
    "hipertensi": "Hipertensión", "hta": "Hipertensión",
    "diabetes": "Diabetes", "diabete": "Diabetes",
    "obesidad": "Obesidad", "obeso": "Obesidad",
    "cardiopatia": "Cardiopatía", "cardiopatía": "Cardiopatía",
    "cardio": "Cardiopatía",
    "epoc": "EPOC",
    "asma": "Asma",
    "anemia": "Anemia",
    "riesgo cardiovascular": "Riesgo Cardiovascular",
    "riesgo cardio": "Riesgo Cardiovascular",
    "cardiovascular": "Riesgo Cardiovascular",
    "normal": "Sin diagnóstico", "sano": "Sin diagnóstico",
    "ninguno": "Sin diagnóstico", "sin diagnostico": "Sin diagnóstico",
    "sin diagnóstico": "Sin diagnóstico", "sin diag": "Sin diagnóstico",
    "ninguna": "Sin diagnóstico", "no aplica": "Sin diagnóstico",
}

# ── Actividad física → categorías ─────────────────────────────
ACTIVIDAD_MAP = {
    "sedentario": "sedentario", "sedentaria": "sedentario",
    "baja": "baja", "ligero": "baja", "ligera": "baja",
    "moderada": "moderada", "moderado": "moderada",
    "alta": "alta", "intensa": "alta", "intenso": "alta",
    "activo": "moderada", "activa": "moderada",
    "ninguna": "sedentario", "nunca": "sedentario",
    "diaria": "alta", "diario": "alta",
    "ocasional": "baja", "ocasionalmente": "baja",
}

# ── IMC Clasificación → categorías ────────────────────────────
IMC_CLASIFICACION_MAP = {
    "bajo_peso": "bajo_peso", "bajo": "bajo_peso",
    "normal": "normal",
    "sobrepeso": "sobrepeso",
    "obesidad": "obesidad", "obeso": "obesidad",
}


# ── Helpers de texto ──────────────────────────────────────────
def _strip_accents(s: str) -> str:
    """Elimina acentos/diacríticos."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _normalize_diag(value):
    """Normaliza diagnóstico a un conjunto canónico de clases."""
    if pd.isna(value) or str(value).strip() == "":
        return "Sin diagnóstico"
    s = _strip_accents(str(value)).strip().lower()
    s_clean = re.sub(r"[^a-z\s]", "", s)
    for key, canonical in DIAG_MAP.items():
        key_norm = _strip_accents(key).lower()
        if key_norm in s_clean:
            return canonical
    return "Sin diagnóstico"


def _normalize_actividad(value):
    """Normaliza actividad_fisica a categorías."""
    if pd.isna(value) or str(value).strip() == "":
        return "desconocida"
    s = _strip_accents(str(value)).strip().lower()
    s_clean = re.sub(r"[^a-z]", "", s)
    return ACTIVIDAD_MAP.get(s_clean, "desconocida")


def _normalize_imc_clasificacion(value):
    """Normaliza IMC clasificación."""
    if pd.isna(value) or str(value).strip() == "":
        return "desconocida"
    s = _strip_accents(str(value)).strip().lower()
    return IMC_CLASIFICACION_MAP.get(s, "desconocida")


# ── Feature Engineering ───────────────────────────────────────
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas: grupos etarios, categorías clínicas,
    interacciones, términos polinomiales, ratios."""
    df = df.copy()

    # --- 1. Codificar sexo (one-hot) ---
    if "sexo" in df.columns:
        sexo_dummies = pd.get_dummies(df["sexo"], prefix="sexo")
        for col in ["sexo_M", "sexo_F"]:
            if col not in sexo_dummies.columns:
                sexo_dummies[col] = 0
        usar = [c for c in ["sexo_M", "sexo_F"] if c in sexo_dummies.columns]
        df = pd.concat([df, sexo_dummies[usar]], axis=1)
        df.drop(columns=["sexo"], inplace=True)

    # --- 2. Codificar actividad_fisica (one-hot) ---
    if "actividad_fisica" in df.columns:
        df["actividad_fisica"] = df["actividad_fisica"].map(_normalize_actividad)
        act_dummies = pd.get_dummies(df["actividad_fisica"], prefix="actividad")
        for col in ["actividad_sedentario", "actividad_baja",
                     "actividad_moderada", "actividad_alta"]:
            if col not in act_dummies.columns:
                act_dummies[col] = 0
        usar_act = [c for c in ["actividad_sedentario", "actividad_baja",
                                 "actividad_moderada", "actividad_alta"]
                    if c in act_dummies.columns]
        df = pd.concat([df, act_dummies[usar_act]], axis=1)
        df.drop(columns=["actividad_fisica"], inplace=True)

    # --- 3. Codificar imc_clasificacion (one-hot) ---
    if "imc_clasificacion" in df.columns:
        df["imc_clasificacion"] = df["imc_clasificacion"].map(_normalize_imc_clasificacion)
        imc_dummies = pd.get_dummies(df["imc_clasificacion"], prefix="imc_clase")
        for col in ["imc_clase_bajo_peso", "imc_clase_normal",
                     "imc_clase_sobrepeso", "imc_clase_obesidad"]:
            if col not in imc_dummies.columns:
                imc_dummies[col] = 0
        usar_imc = [c for c in ["imc_clase_bajo_peso", "imc_clase_normal",
                                 "imc_clase_sobrepeso", "imc_clase_obesidad"]
                    if c in imc_dummies.columns]
        df = pd.concat([df, imc_dummies[usar_imc]], axis=1)
        df.drop(columns=["imc_clasificacion"], inplace=True)

    # --- 4. Codificar riesgo_enfermedad (one-hot) ---
    if "riesgo_enfermedad" in df.columns:
        riesgo_map = {
            "bajo": "bajo", "medio": "medio",
            "alto": "alto", "critico": "critico",
        }
        df["riesgo_enfermedad"] = df["riesgo_enfermedad"].map(
            lambda x: riesgo_map.get(str(x).strip().lower(), "desconocido")
        )
        riesgo_dummies = pd.get_dummies(df["riesgo_enfermedad"], prefix="riesgo")
        for col in ["riesgo_bajo", "riesgo_medio", "riesgo_alto", "riesgo_critico"]:
            if col not in riesgo_dummies.columns:
                riesgo_dummies[col] = 0
        usar_riesgo = [c for c in ["riesgo_bajo", "riesgo_medio",
                                    "riesgo_alto", "riesgo_critico"]
                       if c in riesgo_dummies.columns]
        df = pd.concat([df, riesgo_dummies[usar_riesgo]], axis=1)
        df.drop(columns=["riesgo_enfermedad"], inplace=True)

    # --- 5. Grupos etarios ---
    if "edad" in df.columns:
        df["edad_joven"] = (df["edad"] < 30).astype(int)
        df["edad_adulto"] = ((df["edad"] >= 30) & (df["edad"] < 50)).astype(int)
        df["edad_mayor"] = ((df["edad"] >= 50) & (df["edad"] < 70)).astype(int)
        df["edad_anciano"] = (df["edad"] >= 70).astype(int)

    # --- 6. Categorías de presión arterial ---
    if "presion_sistolica" in df.columns and "presion_diastolica" in df.columns:
        df["hta_normal"] = (
            (df["presion_sistolica"] < 120) & (df["presion_diastolica"] < 80)
        ).astype(int)
        df["hta_prehipertension"] = (
            ((df["presion_sistolica"] >= 120) & (df["presion_sistolica"] < 140)) |
            ((df["presion_diastolica"] >= 80) & (df["presion_diastolica"] < 90))
        ).astype(int)
        df["hta_grado1"] = (
            ((df["presion_sistolica"] >= 140) & (df["presion_sistolica"] < 160)) |
            ((df["presion_diastolica"] >= 90) & (df["presion_diastolica"] < 100))
        ).astype(int)
        df["hta_grado2"] = (
            (df["presion_sistolica"] >= 160) | (df["presion_diastolica"] >= 100)
        ).astype(int)

    # --- 7. Categorías de glucosa ---
    if "glucosa" in df.columns:
        df["glucosa_normal"] = (df["glucosa"] < 100).astype(int)
        df["glucosa_prediabetes"] = ((df["glucosa"] >= 100) & (df["glucosa"] < 126)).astype(int)
        df["glucosa_diabetes"] = (df["glucosa"] >= 126).astype(int)

    # --- 8. Categorías de colesterol ---
    if "colesterol" in df.columns:
        df["colesterol_normal"] = (df["colesterol"] < 200).astype(int)
        df["colesterol_borderline"] = ((df["colesterol"] >= 200) & (df["colesterol"] < 240)).astype(int)
        df["colesterol_alto"] = (df["colesterol"] >= 240).astype(int)

    # --- 9. Interacciones clínicamente relevantes ---
    pares = [
        ("edad", "imc"),
        ("glucosa", "colesterol"),
        ("presion_sistolica", "presion_diastolica"),
        ("imc", "glucosa"),
        ("edad", "presion_sistolica"),
        ("imc", "colesterol"),
        ("edad", "glucosa"),
        ("frecuencia_cardiaca", "presion_sistolica"),
    ]
    for a, b in pares:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]

    # --- 10. Términos cuadráticos (variables clave) ---
    for col in ["edad", "imc", "glucosa", "colesterol", "presion_sistolica"]:
        if col in df.columns:
            df[f"{col}_sq"] = df[col] ** 2

    # --- 11. Ratios clínicos ---
    if "colesterol" in df.columns and "glucosa" in df.columns:
        df["col_glu_ratio"] = df["colesterol"] / (df["glucosa"] + 1e-6)
    if "presion_sistolica" in df.columns and "presion_diastolica" in df.columns:
        df["pulse_pressure"] = df["presion_sistolica"] - df["presion_diastolica"]
    if "peso" in df.columns and "altura" in df.columns:
        df["imc_calc"] = df["peso"] / (df["altura"] ** 2 + 1e-6)
    if "frecuencia_cardiaca" in df.columns and "presion_sistolica" in df.columns:
        df["rate_pressure_product"] = df["frecuencia_cardiaca"] * df["presion_sistolica"]

    return df


def _get_available_features() -> list:
    """Detecta features disponibles en el modelo Paciente."""
    paciente_fields = {f.name for f in Paciente._meta.get_fields()}
    all_raw = FEATURES_NUMERIC + FEATURES_BOOL + FEATURES_CATEGORICAL + FEATURES_DERIVED
    available = [f for f in all_raw if f in paciente_fields]
    if not available:
        raise RuntimeError("No se encontraron features válidos en el modelo Paciente.")
    return available


def _select_features(X_train, y_train, X_test, feature_names, k=40):
    """Selecciona las k features más relevantes usando mutual information."""
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    k = min(k, X_train.shape[1])
    if k == X_train.shape[1]:
        return X_train, X_test, feature_names

    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    mask = selector.get_support()
    selected_features = [f for f, m in zip(feature_names, mask) if m]
    return X_train_sel, X_test_sel, selected_features


def _dataset(features: list) -> pd.DataFrame:
    """Carga pacientes desde la BD, normaliza y aplica feature engineering."""
    qs = Paciente.objects.exclude(
        diagnostico_preliminar=""
    ).exclude(
        diagnostico_preliminar__isnull=True
    ).values(*features, TARGET)

    df = pd.DataFrame(list(qs))
    if df.empty:
        raise RuntimeError("No hay pacientes con diagnóstico en la base de datos.")

    # Normalizar target
    df[TARGET] = df[TARGET].apply(_normalize_diag)

    # Eliminar clases con menos de 5 ejemplos
    class_counts = df[TARGET].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df[TARGET].isin(valid_classes)]
    if df.empty:
        raise RuntimeError("No hay suficientes ejemplos por clase después de filtrar.")

    # Booleanos → int
    for c in FEATURES_BOOL:
        if c in df.columns:
            df[c] = df[c].astype(int)

    # Imputar numéricas con mediana
    for c in features:
        if c in df.columns and c not in FEATURES_CATEGORICAL and c not in FEATURES_DERIVED and df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Imputar categóricas con moda
    for c in FEATURES_CATEGORICAL + FEATURES_DERIVED:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(
                df[c].mode()[0] if not df[c].mode().empty else "desconocida"
            )

    df = _engineer_features(df)
    return df


# ── ENTRENAMIENTO ─────────────────────────────────────────────
def train() -> ModelMetrics:
    """Entrena RandomForest optimizado con feature engineering,
    selección de características, SMOTE y tuning de hiperparámetros."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
    from sklearn.model_selection import train_test_split, RandomizedSearchCV

    features = _get_available_features()
    df = _dataset(features)

    if len(df) < 50:
        raise RuntimeError(
            f"Solo hay {len(df)} registros válidos. Se necesitan al menos 50."
        )

    # Columnas del feature engineering
    final_raw = [f for f in features if f in df.columns]
    target_set = {TARGET}
    engineered = [c for c in df.columns if c not in set(final_raw) | target_set]
    model_features = final_raw + engineered

    X = df[model_features].values
    y = df[TARGET].values

    # ── Train/test split estratificado ──
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    # ── Escalar (estandariza, no afecta a RF pero sí a feature select) ──
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # ── Selección de features con mutual information ──
    unique_classes = np.unique(y_tr)
    n_features_orig = X_tr_scaled.shape[1]
    k_features = min(45, n_features_orig, max(20, len(unique_classes) * 4))
    X_tr_sel, X_te_sel, selected_features = _select_features(
        X_tr_scaled, y_tr, X_te_scaled, model_features, k=k_features
    )
    n_features_sel = X_tr_sel.shape[1]

    # ── SMOTE (balanceo de clases) ──
    use_smote = False
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore[import-untyped]
        n_neighbors = min(5, len(np.unique(y_tr)) - 1)
        if n_neighbors >= 2:
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
            try:
                X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sel, y_tr)
                use_smote = True
            except Exception:
                X_tr_bal, y_tr_bal = X_tr_sel, y_tr
        else:
            X_tr_bal, y_tr_bal = X_tr_sel, y_tr
    except ImportError:
        X_tr_bal, y_tr_bal = X_tr_sel, y_tr

    # ── RandomForest con class_weight para balance adicional ──
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1,
        class_weight="balanced" if not use_smote else None,
    )

    # ── Búsqueda de hiperparámetros ──
    param_dist = {
        "n_estimators": [200, 300, 500, 800, 1000],
        "max_depth": [10, 15, 20, 25, 30, None],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["gini", "entropy"],
        "bootstrap": [True, False],
    }

    n_iter = min(50, len(param_dist) * 3)
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="f1_weighted",
        random_state=42,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_tr_bal, y_tr_bal)
    best_rf = search.best_estimator_
    y_pred = best_rf.predict(X_te_sel)
    classes = sorted(np.unique(y).tolist())

    best_params = search.best_params_
    best_params["use_smote"] = use_smote
    best_params["n_features_original"] = n_features_orig
    best_params["n_features_selected"] = n_features_sel

    # Feature importances
    feature_importances = {}
    if hasattr(best_rf, "feature_importances_"):
        fi = best_rf.feature_importances_
        if len(fi) == len(selected_features):
            for i, fname in enumerate(selected_features):
                feature_importances[fname] = float(fi[i])

    metrics = ModelMetrics.objects.create(
        accuracy=float(accuracy_score(y_te, y_pred)),
        precision=float(precision_score(
            y_te, y_pred, average="weighted", zero_division=0
        )),
        recall=float(recall_score(
            y_te, y_pred, average="weighted", zero_division=0
        )),
        f1=float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
        confusion_matrix=confusion_matrix(
            y_te, y_pred, labels=classes
        ).tolist(),
        classes=classes,
        feature_importances=feature_importances,
        n_samples=len(df),
    )

    joblib.dump({
        "model": best_rf,
        "scaler": scaler,
        "features": selected_features,
        "classes": classes,
        "best_params": best_params,
    }, MODEL_PATH)
    return metrics


# ── PREDICCIÓN ────────────────────────────────────────────────
def predict(paciente_id: int) -> dict:
    """Predice el diagnóstico de un paciente usando el modelo entrenado."""
    if not MODEL_PATH.exists():
        raise ValueError(
            "El modelo no ha sido entrenado aún. "
            "Ejecuta POST /api/predicciones/train/ primero."
        )

    artifact = joblib.load(MODEL_PATH)
    clf = artifact["model"]
    scaler = artifact["scaler"]
    classes = artifact["classes"]
    model_features = artifact["features"]

    try:
        paciente = Paciente.objects.get(pk=paciente_id)
    except Paciente.DoesNotExist:
        raise ValueError(f"Paciente con id {paciente_id} no encontrado.")

    # Construir vector de features raw desde el objeto Paciente
    raw_features = _get_available_features()
    row: dict[str, float | int | str] = {}
    for f in raw_features:
        val = getattr(paciente, f, None)
        if val is None:
            val = 0
        if f in FEATURES_BOOL:
            row[f] = int(val)
        elif f in FEATURES_CATEGORICAL or f in FEATURES_DERIVED:
            row[f] = str(val) if val else ""
        else:
            row[f] = float(val) if val else 0.0

    df_pred = pd.DataFrame([row])
    df_pred = _engineer_features(df_pred)

    # Rellenar columnas faltantes con 0
    for f in model_features:
        if f not in df_pred.columns:
            df_pred[f] = 0

    X = df_pred[model_features].values
    X_scaled = scaler.transform(X)

    proba = clf.predict_proba(X_scaled)[0]
    top_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {"diagnostico": classes[i], "confianza": round(float(proba[i]) * 100, 1)}
        for i in top_idx
        if proba[i] > 0
    ]

    return {
        "paciente_id": paciente_id,
        "nombre": f"{paciente.nombres} {paciente.apellidos}",
        "diagnostico": top3[0]["diagnostico"],
        "confianza": top3[0]["confianza"],
        "top3": top3,
    }
