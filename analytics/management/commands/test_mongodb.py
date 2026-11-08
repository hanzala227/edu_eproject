from django.core.management.base import BaseCommand
from django.conf import settings
from edupredict.db import check_mongo_connection, get_db


class Command(BaseCommand):
    help = 'Test connection to the MongoDB database configured in settings/.env'

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Testing connection to MongoDB..."))
        
        uri = getattr(settings, 'MONGODB_URI', '')
        if not uri:
            self.stdout.write(
                self.style.WARNING(
                    "[!] MONGODB_URI is empty. Please paste your MongoDB connection string in the .env file."
                )
            )
            return

        # Obfuscate credentials when printing URI
        display_uri = uri
        if '@' in uri:
            prefix = uri.split('@')[0]
            host_part = uri.split('@')[1]
            scheme = prefix.split('://')[0]
            display_uri = f"{scheme}://****:****@{host_part}"

        self.stdout.write(f"Configured URI: {display_uri}")
        self.stdout.write(f"Database Name: {settings.MONGODB_DB_NAME}")

        success, message = check_mongo_connection()
        if success:
            self.stdout.write(self.style.SUCCESS(f"[OK] {message}"))
            try:
                db = get_db()
                collections = db.list_collection_names()
                self.stdout.write(f"Existing Collections ({len(collections)}): {collections}")
            except Exception as e:
                self.stdout.write(self.style.WARNING(f"Could not list collections: {e}"))
        else:
            self.stdout.write(self.style.ERROR(f"[FAIL] {message}"))
