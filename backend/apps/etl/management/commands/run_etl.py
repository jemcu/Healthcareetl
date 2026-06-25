from django.core.management.base import BaseCommand
from apps.etl.engine import run_etl


class Command(BaseCommand):
    help = "Ejecuta el ETL completo sobre un archivo CSV o XLSX."

    def add_arguments(self, parser):
        parser.add_argument("path", help="Ruta al archivo CSV/XLSX")

    def handle(self, path, **kwargs):
        run = run_etl(path)
        self.stdout.write(self.style.SUCCESS(
            f"ETL {run.status} · {run.rows_loaded} cargados · "
            f"{run.duplicates_removed} duplicados · {run.outliers_fixed} outliers · "
            f"{run.duration_ms}ms"
        ))
        self.stdout.write(run.log)
