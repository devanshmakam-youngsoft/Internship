import json
import re

def extract_json(text, fallback=None):
    if not text:
        return fallback

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return fallback
