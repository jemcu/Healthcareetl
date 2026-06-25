from rest_framework.views import APIView
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiResponse
from . import services

@extend_schema(
    tags=["Analytics"],
    responses={200: OpenApiResponse(description="KPIs y distribuciones del dashboard")},
)
class KPIsView(APIView):
    def get(self, request):
        return Response({
            "kpis": services.kpis(),
            "riesgo": services.distribucion_riesgo(),
            "sexo": services.distribucion_sexo(),
            "imc": services.distribucion_imc(),
            "diagnosticos": services.top_diagnosticos(),
            "edad": services.segmentacion_edad(),
            "criticos": services.pacientes_criticos(),
        })

@extend_schema(
    tags=["Analytics"],
    responses={200: OpenApiResponse(description="Estadística descriptiva de variables clínicas")},
)
class StatsView(APIView):
    def get(self, request):
        return Response(services.estadistica_descriptiva())
