"""
EduPredict - MongoDB Connection & Collection Manager
Provides a singleton MongoClient and helper utilities for database operations.
"""

import logging
from django.conf import settings
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

try:
    import certifi
    CA_FILE = certifi.where()
except Exception:
    CA_FILE = None

# Ensure robust DNS SRV resolution for MongoDB Atlas on Linux
try:
    import dns.resolver
    resolver = dns.resolver.Resolver()
    # Fallback to public DNS servers if local systemd-resolved struggles with SRV
    resolver.nameservers = ['8.8.8.8', '1.1.1.1'] + resolver.nameservers
    dns.resolver.default_resolver = resolver
except Exception:
    pass

logger = logging.getLogger(__name__)

# Standard collection names
COLL_USERS = 'users'
COLL_STUDENTS = 'students'
COLL_ATTENDANCE = 'attendance'
COLL_ACADEMIC_RECORDS = 'academic_records'
COLL_LMS_ACTIVITY = 'lms_activity'
COLL_PREDICTIONS = 'predictions'
COLL_ALERTS = 'alerts'
COLL_COURSES = 'courses'
COLL_FEEDBACK = 'feedback'
COLL_SYSTEM_LOGS = 'system_logs'


class MongoDBManager:
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        uri = getattr(settings, 'MONGODB_URI', None)
        db_name = getattr(settings, 'MONGODB_DB_NAME', 'edupredict_db')

        if not uri or not uri.strip():
            logger.warning("MONGODB_URI is not set in settings or .env file.")
            self._client = None
            self._db = None
            return

        try:
            client_kwargs = {
                'serverSelectionTimeoutMS': 7000,
                'connectTimeoutMS': 7000,
            }
            if CA_FILE:
                client_kwargs['tlsCAFile'] = CA_FILE

            self._client = MongoClient(uri, **client_kwargs)
            self._db = self._client[db_name]
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB client: {e}")
            self._client = None
            self._db = None

    def get_client(self):
        if self._client is None:
            self._init_connection()
        return self._client

    def get_db(self):
        if self._db is None:
            self._init_connection()
        return self._db

    def get_collection(self, collection_name: str):
        db = self.get_db()
        if db is None:
            raise ConnectionError(
                "MongoDB connection is not established. Please check your MONGODB_URI."
            )
        return db[collection_name]

    def test_connection(self):
        """
        Tests the connection by pinging the database.
        Returns (success: bool, message: str)
        """
        client = self.get_client()
        if not client:
            return False, "MONGODB_URI is missing or not configured."

        try:
            client.admin.command('ping')
            db_name = getattr(settings, 'MONGODB_DB_NAME', 'edupredict_db')
            return True, f"Successfully connected to MongoDB Atlas! Database: '{db_name}'"
        except ServerSelectionTimeoutError as e:
            return False, (
                "Connection timed out. In MongoDB Atlas, your current IP address needs to be whitelisted under "
                "'Network Access' -> 'Add IP Address' (or allow '0.0.0.0/0' for access from anywhere)."
            )
        except ConnectionFailure as e:
            return False, f"MongoDB connection failure (check username/password): {e}"
        except Exception as e:
            return False, f"Error connecting to MongoDB: {e}"


# Singleton instance helper functions
mongo_manager = MongoDBManager()


def get_db():
    """Retrieve the active MongoDB database object."""
    return mongo_manager.get_db()


def get_collection(name: str):
    """Retrieve a MongoDB collection by name."""
    return mongo_manager.get_collection(name)


def check_mongo_connection():
    """Check if the connection to MongoDB is alive."""
    return mongo_manager.test_connection()
