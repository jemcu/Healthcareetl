from django.db import models
from django.conf import settings


class Paciente(models.Model):
    SEXO = [("M", "Masculino"), ("F", "Femenino"), ("O", "Otro")]
    RIESGO = [("bajo", "Bajo"), ("medio", "Medio"), ("alto", "Alto"), ("critico", "Crítico")]
    IMC_CLASS = [
        ("bajo_peso", "Bajo peso"), ("normal", "Normal"),
        ("sobrepeso", "Sobrepeso"), ("obesidad", "Obesidad"),
    ]

    id_paciente = models.IntegerField(unique=True)
    nombres = models.CharField(max_length=120)
    apellidos = models.CharField(max_length=120)
    edad = models.IntegerField()
    sexo = models.CharField(max_length=1, choices=SEXO)
    peso = models.FloatField()
    altura = models.FloatField()
    imc = models.FloatField()
    imc_clasificacion = models.CharField(max_length=20, choices=IMC_CLASS)
    presion_sistolica = models.IntegerField()
    presion_diastolica = models.IntegerField()
    frecuencia_cardiaca = models.IntegerField()
    glucosa = models.FloatField()
    colesterol = models.FloatField()
    saturacion_oxigeno = models.FloatField()
    temperatura = models.FloatField()
    antecedentes_familiares = models.BooleanField(default=False)
    fumador = models.BooleanField(default=False)
    consumo_alcohol = models.BooleanField(default=False)
    actividad_fisica = models.CharField(max_length=30, blank=True)
    diagnostico_preliminar = models.CharField(max_length=120, blank=True)
    riesgo_enfermedad = models.CharField(max_length=10, choices=RIESGO)
    fecha_consulta = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.id_paciente} · {self.nombres} {self.apellidos}"


class ETLRun(models.Model):
    STATUS = [("ok", "OK"), ("error", "Error"), ("partial", "Parcial")]

    source = models.CharField(max_length=255)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                             on_delete=models.SET_NULL)
    started_at = models.DateTimeField(auto_now_add=True)
    duration_ms = models.IntegerField(default=0)
    rows_extracted = models.IntegerField(default=0)
    rows_after_transform = models.IntegerField(default=0)
    rows_loaded = models.IntegerField(default=0)
    duplicates_removed = models.IntegerField(default=0)
    nulls_imputed = models.IntegerField(default=0)
    outliers_fixed = models.IntegerField(default=0)
    status = models.CharField(max_length=10, choices=STATUS, default="ok")
    log = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]
