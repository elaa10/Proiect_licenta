import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://phishing_user:phishing_pass@db:5432/phishing_db")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "schimba_asta")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))

settings = Settings()