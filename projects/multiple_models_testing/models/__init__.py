# models/__init__.py
from core.registry import register_model

# REGISTER MODELS

#dummy
from models.dummy import DummyModel
register_model("dummy-model",DummyModel)

#gemini
from models.gemini import Gemini25Model
register_model("gemini-2.5",Gemini25Model)

#llama3
from models.hf_llama3_api import HFLlama3Model
register_model("hf-llama3-8b",HFLlama3Model)