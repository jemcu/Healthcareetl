from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

class Command(BaseCommand):
    help = "Crea los usuarios por defecto si no existen"

    def handle(self, *args, **kwargs):
        User = get_user_model()

        usuarios = [
            {"username": "admin",    "password": "admin123",    "email": "admin@healthanalytics.com",    "is_superuser": True,  "is_staff": True,  "role": User.Role.ADMIN},
            {"username": "medico",   "password": "medico123",   "email": "medico@healthanalytics.com",   "is_superuser": False, "is_staff": False, "role": User.Role.MEDICO},
            {"username": "analista", "password": "analista123", "email": "analista@healthanalytics.com", "is_superuser": False, "is_staff": False, "role": User.Role.ANALISTA},
        ]

        for u in usuarios:
            if User.objects.filter(username=u["username"]).exists():
                self.stdout.write(f"Usuario '{u['username']}' ya existe.")
                continue

            user = User.objects.create_superuser(
                username=u["username"],
                email=u["email"],
                password=u["password"],
            ) if u["is_superuser"] else User.objects.create_user(
                username=u["username"],
                email=u["email"],
                password=u["password"],
            )
            user.is_staff = u["is_staff"]
            user.role = u["role"]
            user.save()
            self.stdout.write(self.style.SUCCESS(f"✓ Usuario '{u['username']}' creado con role='{u['role']}'."))
