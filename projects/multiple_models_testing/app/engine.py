from core.registry import load_model

class ChatEngine:
    def __init__(self, model_name: str):
        self.model = load_model(model_name)
        self.model_name= model_name

    def chat(self):
        print("\nType 'exit' to quit")

        while True:
            prompt = input("\nYou: ")
            if prompt.lower() == "exit":
                break

            response = self.model.generate(prompt)
            print(f"\n{self.model_name}: {response}")
