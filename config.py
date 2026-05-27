import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# DashScope OpenAI-compatible endpoint
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ChromaDB persist directory
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma")
