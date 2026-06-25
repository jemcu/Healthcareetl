from django.core.management.base import BaseCommand
from apps.ml.engine import train


class Command(BaseCommand):
    help = "Entrena el modelo Random Forest sobre los pacientes cargados."

    def handle(self, *a, **k):
        m = train()
        self.stdout.write(self.style.SUCCESS(
            f"Modelo entrenado. accuracy={m.accuracy:.3f} f1={m.f1:.3f} n={m.n_samples}"
        ))
