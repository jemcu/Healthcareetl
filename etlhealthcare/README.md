# HealthAnalytics IPS — Plataforma Inteligente de Analítica Clínica

Aplicación FullStack para procesamiento ETL, analítica de datos, KPIs clínicos y predicción de riesgo médico con Machine Learning.

**Stack:** Python 3.12 · Django 5 · Django REST Framework · Pandas · NumPy · Scikit-Learn · PostgreSQL · Bootstrap 5 · Chart.js · JWT.

---

## 1. Arquitectura

```
healthcare-etl-platform/
├── backend/
│   ├── config/                 # Settings, urls, wsgi
│   ├── apps/
│   │   ├── authentication/     # Custom user + roles + JWT
│   │   ├── etl/                # Extract / Transform / Load (pandas)
│   │   ├── analytics/          # KPIs y estadística descriptiva
│   │   ├── ml/                 # Random Forest + métricas
│   │   ├── dashboard/          # Vistas server-rendered
│   │   └── reports/            # Exportación CSV / Excel / PDF
│   ├── templates/              # Bootstrap 5
│   ├── static/                 # CSS / JS / Chart.js
│   ├── manage.py
│   └── requirements.txt
├── datasets/                   # Dataset clínico simulado (1800 registros)
├── docs/
├── docker/
├── render.yaml                 # Despliegue Render
└── README.md
```

## 2. Instalación local

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # editar credenciales
python manage.py migrate
python manage.py seed_users        # crea admin/medico/analista demo
python manage.py runserver
```

Abre <http://localhost:8000>. Credenciales demo:

| Rol           | Usuario   | Contraseña   |
|---------------|-----------|--------------|
| Administrador | `admin`   | `Admin12345` |
| Médico        | `medico`  | `Medico12345`|
| Analista      | `analista`| `Analista12345` |

## 3. Ejecutar el ETL sobre el dataset entregado

```bash
python manage.py run_etl ../datasets/dataset_clinico_etl_1800_registros.xlsx
```

El comando:

1. **EXTRACT** — lee el archivo (xlsx/csv), registra fuente y tiempo.
2. **TRANSFORM** — elimina duplicados, corrige tipos, normaliza diagnósticos,
   imputa nulos (media/mediana/moda), recalcula IMC, valida rangos clínicos,
   clasifica riesgo (Bajo / Medio / Alto / Crítico).
3. **LOAD** — inserta en `pacientes`, registra `ETLRun` con trazabilidad.

También puedes subir un CSV desde la UI en *ETL → Cargar dataset*.

## 4. Machine Learning

```bash
python manage.py train_model
```

Entrena un **Random Forest** sobre los pacientes ya cargados, guarda el
modelo en `backend/apps/ml/artifacts/model.pkl` y persiste **Accuracy,
Precision, Recall, F1, matriz de confusión** en BD. Visibles en *ML →
Métricas*.

Predicción individual:

```
POST /api/predicciones/   (JWT)
{ "edad": 58, "imc": 31.2, "glucosa": 180, "colesterol": 240,
  "presion_sistolica": 150, "frecuencia_cardiaca": 92, "fumador": true }
```

## 5. APIs REST principales

| Método | Endpoint                  | Descripción                       |
|--------|---------------------------|-----------------------------------|
| POST   | `/api/auth/login/`        | Obtiene JWT                       |
| POST   | `/api/auth/refresh/`      | Refresca JWT                      |
| GET    | `/api/pacientes/`         | Lista paginada (filtros)          |
| POST   | `/api/etl/run/`           | Ejecuta ETL sobre archivo subido  |
| GET    | `/api/etl/history/`       | Historial de ejecuciones          |
| GET    | `/api/dashboard/kpis/`    | KPIs clínicos                     |
| GET    | `/api/analytics/stats/`   | Estadística descriptiva           |
| POST   | `/api/predicciones/`      | Predicción de riesgo individual   |
| GET    | `/api/reportes/<tipo>/`   | Exporta `csv` / `xlsx` / `pdf`    |

Swagger: <http://localhost:8000/api/docs/>

## 6. Despliegue en Render

1. Sube el repo a GitHub.
2. En Render → **New → Blueprint** → selecciona el repo.
3. Render detecta `render.yaml` y crea:
   - Servicio web Django (gunicorn + whitenoise).
   - Base de datos PostgreSQL gestionada.
4. Variables generadas automáticamente:
   `DATABASE_URL`, `SECRET_KEY`, `DJANGO_DEBUG=False`, `ALLOWED_HOSTS=*`.
5. En el primer deploy se ejecuta `release` (`migrate` + `seed_users`).

## 6.1 Despliegue en Vercel

El proyecto incluye `vercel.json`, `api/index.py` y `requirements.txt` en la
raiz para desplegar Django como funcion serverless.

Variables recomendadas en Vercel:

| Variable | Valor |
|----------|-------|
| `SECRET_KEY` | Clave secreta fuerte |
| `DJANGO_DEBUG` | `False` |
| `DATABASE_URL` | URL PostgreSQL externa |
| `ALLOWED_HOSTS` | `.vercel.app,localhost,127.0.0.1` |
| `CSRF_TRUSTED_ORIGINS` | `https://*.vercel.app` |

Antes o despues del primer deploy ejecuta las migraciones contra la misma
`DATABASE_URL`:

```bash
cd backend
python manage.py migrate
python manage.py seed_users
```

## 7. Seguridad

- JWT (SimpleJWT) + permisos por rol (`IsAdmin`, `IsMedico`, `IsAnalista`).
- CSRF activo en vistas HTML.
- Sanitización de inputs (DRF serializers + validadores).
- Variables sensibles en `.env`.

## 8. Criterios cumplidos

- ✅ ETL completo con logs y trazabilidad.
- ✅ Limpieza, normalización, tratamiento de nulos, validación clínica.
- ✅ Clasificación de riesgo + cálculo IMC + clasificación clínica.
- ✅ Random Forest con métricas + matriz de confusión.
- ✅ KPIs, segmentación, detección de críticos.
- ✅ Dashboard Bootstrap 5 + Chart.js (barras, líneas, pie, heatmap, tendencias).
- ✅ Auth + roles + JWT.
- ✅ Subida manual CSV + exportación CSV/Excel/PDF.
- ✅ Historial ETL.
- ✅ Listo para Render (PostgreSQL gestionado).
