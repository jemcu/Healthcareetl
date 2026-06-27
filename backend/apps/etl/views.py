from rest_framework import viewsets, status, parsers
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema, OpenApiResponse
from apps.authentication.permissions import IsAnalistaOrAdmin
from .models import Paciente, ETLRun
from .serializers import PacienteSerializer, ETLRunSerializer
from .engine import run_etl

class PacienteViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Paciente.objects.all()
    serializer_class = PacienteSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["sexo", "riesgo_enfermedad", "imc_clasificacion", "fumador"]

@extend_schema(
    tags=["ETL"],
    request={"multipart/form-data": {"type": "object", "properties": {"file": {"type": "string", "format": "binary"}}}},
    responses={
        201: ETLRunSerializer,
        400: OpenApiResponse(description="Archivo requerido"),
    },
)
class ETLRunView(APIView):
    permission_classes = [IsAnalistaOrAdmin]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request):
        f = request.FILES.get("file")
        if not f:
            return Response({"detail": "Archivo requerido (campo 'file')."},
                            status=status.HTTP_400_BAD_REQUEST)
        run = run_etl(f, user=request.user)
        return Response(ETLRunSerializer(run).data, status=status.HTTP_201_CREATED)

@extend_schema(
    tags=["ETL"],
    responses={200: ETLRunSerializer(many=True)},
)
class ETLHistoryView(APIView):
    def get(self, request):
        qs = ETLRun.objects.all()[:50]
        return Response(ETLRunSerializer(qs, many=True).data)
