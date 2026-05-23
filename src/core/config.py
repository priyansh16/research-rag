import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "RAG Assistant",
    DEBUG:bool = os.getenv("DEBUG", "True") == "True"
    

settings = Settings()
