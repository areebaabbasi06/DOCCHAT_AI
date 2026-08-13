
import os
from dotenv import load_dotenv

load_dotenv()


# =====================================
# Qdrant Cloud Configuration
# =====================================

QDRANT_URL = os.getenv("QDRANT_URL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")


# =====================================
# PostgreSQL Configuration
# =====================================

DB_HOST = os.getenv("DB_HOST")

DB_PORT = int(
    os.getenv(
        "DB_PORT",
        5432
    )
)

DB_NAME = os.getenv("DB_NAME")

DB_USER = os.getenv("DB_USER")

DB_PASSWORD = os.getenv("DB_PASSWORD")


# =====================================
# Gemini API Configuration
# =====================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

