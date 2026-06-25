from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiResponse
from apps.authentication.permissions import IsAnalistaOrAdmin
from .engine import train, predict
from .models import ModelMetrics

@extend_schema(
    tags=["ML"],
    request=None,
    responses={
        200: OpenApiResponse(description="Métricas del modelo recién entrenado"),
        400: OpenApiResponse(description="Error durante el entrenamiento"),
    },
)
class TrainView(APIView):
    permission_classes = [IsAnalistaOrAdmin]

    def post(self, request):
        try:
            m = train()
        except Exception as exc:
            return Response({"detail": str(exc)}, status=400)
        return Response({
            "accuracy": m.accuracy, "precision": m.precision,
            "recall": m.recall, "f1": m.f1,
            "confusion_matrix": m.confusion_matrix, "classes": m.classes,
            "feature_importances": m.feature_importances, "n_samples": m.n_samples,
        })

@extend_schema(
    tags=["ML"],
    responses={
        200: OpenApiResponse(description="Métricas del modelo actualmente entrenado"),
        404: OpenApiResponse(description="Modelo no entrenado aún"),
    },
)
class MetricsView(APIView):
    def get(self, request):
        m = ModelMetrics.objects.first()
        if not m:
            return Response({"detail": "Modelo no entrenado."}, status=404)
        return Response({
            "accuracy": m.accuracy, "precision": m.precision,
            "recall": m.recall, "f1": m.f1,
            "confusion_matrix": m.confusion_matrix, "classes": m.classes,
            "feature_importances": m.feature_importances,
            "n_samples": m.n_samples, "trained_at": m.trained_at,
        })

@extend_schema(
    tags=["ML"],
    request=OpenApiResponse(description="Datos clínicos del paciente en JSON"),
    responses={
        200: OpenApiResponse(description="Predicción de riesgo"),
        400: OpenApiResponse(description="Error en los datos de entrada"),
    },
)
class PredictView(APIView):
    def post(self, request):
        try:
            return Response(predict(request.data))
        except Exception as exc:
            return Response({"detail": str(exc)}, status=400)
