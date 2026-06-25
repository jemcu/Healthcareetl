from rest_framework import serializers
from .models import Paciente, ETLRun


class PacienteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Paciente
        fields = "__all__"


class ETLRunSerializer(serializers.ModelSerializer):
    user_name = serializers.CharField(source="user.username", read_only=True, default="")

    class Meta:
        model = ETLRun
        fields = "__all__"
