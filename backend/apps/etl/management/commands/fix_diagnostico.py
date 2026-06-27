"""
Normaliza los valores de diagnostico_preliminar en la BD.
Idempotente: los valores que no coinciden con el mapa se
dejan intactos para poder ejecutarlo múltiples veces sin efectos laterales.
"""
from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction

from apps.etl.diagnosis import CANONICAL_VALUES, DIAG_MAP, normalize
from apps.etl.models import Paciente


class Command(BaseCommand):
    help = "Corrige encoding de diagnostico_preliminar en todos los pacientes"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Solo mostrar lo que se corregiría sin escribir en BD.",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Mostrar cada cambio individual (anterior → nuevo).",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        verbose = options["verbose"]

        pacientes = Paciente.objects.exclude(
            diagnostico_preliminar__isnull=True
        ).exclude(diagnostico_preliminar="")

        total = pacientes.count()
        if total == 0:
            self.stdout.write("No hay pacientes con diagnóstico.")
            return

        cambios: list[tuple[int, str, str]] = []  # (id_paciente, old, new)
        desconocidos: dict[str, int] = {}  # valor_original → count

        for p in pacientes.iterator():
            nuevo = normalize(p.diagnostico_preliminar)
            if nuevo == p.diagnostico_preliminar:
                # Valor ya normalizado — ver si es conocido o no
                if nuevo not in CANONICAL_VALUES:
                    desconocidos[nuevo] = desconocidos.get(nuevo, 0) + 1
                continue

            cambios.append((p.id_paciente, p.diagnostico_preliminar, nuevo))

            if nuevo not in CANONICAL_VALUES:
                desconocidos[nuevo] = desconocidos.get(nuevo, 0) + 1

        n_corregibles = len(cambios)
        n_ya_normalizados = total - n_corregibles

        # ── Reporte resumen ──────────────────────────────────────────
        self.stdout.write(self.style.MIGRATE_HEADING("═══ Resumen ═══"))
        self.stdout.write(f"  Total pacientes con diagnóstico: {total}")
        self.stdout.write(f"  Ya normalizados:                 {n_ya_normalizados}")

        if n_corregibles > 0:
            style = self.style.WARNING if dry_run else self.style.SUCCESS
            self.stdout.write(style(
                f"  {'Se corregirían' if dry_run else 'Corregidos'}: {n_corregibles}"
            ))
        else:
            self.stdout.write(self.style.SUCCESS("  Corregidos: 0 (todo OK)"))

        # ── Verbose: listar cambios ───────────────────────────────────
        if verbose and cambios:
            self.stdout.write("")
            self.stdout.write(self.style.MIGRATE_HEADING("═══ Cambios ═══"))
            for pid, old, new in cambios:
                self.stdout.write(f"  Paciente #{pid}: {old!r} → {new!r}")

        # ── Valores desconocidos ──────────────────────────────────────
        if desconocidos:
            self.stdout.write("")
            self.stdout.write(self.style.WARNING("═══ Valores no reconocidos ═══"))
            self.stdout.write(
                "  Estos valores no están en DIAG_MAP. Si alguno debería "
                "normalizarse, agrégalo a DIAG_MAP en apps/etl/diagnosis.py."
            )
            for val, count in sorted(desconocidos.items(), key=lambda x: -x[1]):
                self.stdout.write(f"  • {val!r}: {count} paciente(s)")

        # ── Ejecutar en BD (solo si no es dry-run) ────────────────────
        if not dry_run and cambios:
            self._apply_changes(cambios)

    @transaction.atomic
    def _apply_changes(self, cambios: list[tuple[int, str, str]]) -> None:
        """Aplica los cambios en la BD usando bulk_update para eficiencia."""
        ids = [c[0] for c in cambios]
        old_to_new = {(c[0]): c[2] for c in cambios}

        pacientes = Paciente.objects.filter(id_paciente__in=ids)
        for p in pacientes:
            p.diagnostico_preliminar = old_to_new[p.id_paciente]

        Paciente.objects.bulk_update(pacientes, ["diagnostico_preliminar"])
