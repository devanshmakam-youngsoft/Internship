import os
import models  # ensures models are registered
from dotenv import load_dotenv

from core.registry import list_models
from app.engine import ChatEngine

# Load environment variables
load_dotenv()

def main():
    models_available = list_models()

    # Get model name from .env
    model_name = os.getenv("MODEL_USE")

    if not model_name:
        raise ValueError("MODEL_USE not set in .env")

    if model_name not in models_available:
        raise ValueError(
            f"MODEL_USE='{model_name}' is invalid.\n"
            f"Available models: {models_available}"
        )

    print(f"Using model from .env → {model_name}")

    engine = ChatEngine(model_name)
    engine.chat()

if __name__ == "__main__":
    main()
