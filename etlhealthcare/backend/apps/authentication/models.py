from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Role(models.TextChoices):
        ADMIN = "admin", "Administrador"
        MEDICO = "medico", "Médico"
        ANALISTA = "analista", "Analista"

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.ANALISTA)

    @property
    def is_admin(self): return self.role == self.Role.ADMIN

    @property
    def is_medico(self): return self.role == self.Role.MEDICO

    @property
    def is_analista(self): return self.role == self.Role.ANALISTA
