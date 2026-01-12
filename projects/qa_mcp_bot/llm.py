from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
import time
from config import HF_API_KEY

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

client = InferenceClient(model=MODEL, token=HF_API_KEY)

SYSTEM_PROMPT = (
    "You are a strict JSON-only assistant. "
    "Do not explain. Do not add markdown. "
    "Return ONLY valid JSON."
)

def call_llm(prompt, retries=1):
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.2
            )
            return response.choices[0].message.content

        except HfHubHTTPError as e:
            if e.response and e.response.status_code == 429:
                if attempt < retries:
                    time.sleep(20)
                    continue
                return None
            raise e
