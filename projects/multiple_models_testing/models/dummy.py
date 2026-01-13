from core.base_model import BaseModel
from core.registry import register_model

class DummyModel(BaseModel):

    def load(self):
        print("Dummy model loaded")

    def generate(self, prompt: str) -> str:
        return f"[Dummy reply to]: {prompt}"

