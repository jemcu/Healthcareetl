from django.contrib import admin
from .models import Paciente, ETLRun
admin.site.register(Paciente)
admin.site.register(ETLRun)
