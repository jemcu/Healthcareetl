from django.core.management.base import BaseCommand
from apps.authentication.models import User


class Command(BaseCommand):
    help = "Crea usuarios demo (admin, medico, analista)."

    def handle(self, *args, **kwargs):
        seeds = [
            ("admin",   "Admin12345",    "admin",    True,  True),
            ("medico",  "Medico12345",   "medico",   False, False),
            ("analista","Analista12345", "analista", False, False),
        ]
        for username, pwd, role, is_staff, is_superuser in seeds:
            u, created = User.objects.get_or_create(
                username=username,
                defaults={"role": role, "is_staff": is_staff, "is_superuser": is_superuser},
            )
            u.role = role
            u.is_staff = is_staff
            u.is_superuser = is_superuser
            u.set_password(pwd)
            u.save()
            self.stdout.write(self.style.SUCCESS(
                f"{'Creado' if created else 'Actualizado'}: {username} / {pwd} ({role})"
            ))
