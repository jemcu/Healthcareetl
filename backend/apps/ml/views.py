from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from drf_spectacular.utils import extend_schema, OpenApiResponse
from apps.authentication.permissions import IsAnalistaOrAdmin
from .engine import train, predict
from .models import ModelMetrics


@extend_schema(tags=["ML"])
class TrainView(APIView):
    permission_classes = [IsAnalistaOrAdmin]

    def post(self, request):
        try:
            m = train()
        except Exception as exc:
            return Response({"detail": str(exc)}, status=400)
        return Response({
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "confusion_matrix": m.confusion_matrix,
            "classes": m.classes,
            "feature_importances": m.feature_importances,
            "n_samples": m.n_samples,
        })


@extend_schema(tags=["ML"])
class MetricsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        m = ModelMetrics.objects.first()
        if not m:
            return Response({"detail": "Modelo no entrenado."}, status=404)
        return Response({
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "confusion_matrix": m.confusion_matrix,
            "classes": m.classes,
            "feature_importances": m.feature_importances,
            "n_samples": m.n_samples,
            "trained_at": m.trained_at,
        })


@extend_schema(tags=["ML"])
class PredictView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, paciente_id):
        try:
            result = predict(paciente_id)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=404)
        except Exception as exc:
            return Response({"detail": str(exc)}, status=400)
        return Response(result)
