import json
import os
from config import CHAT_HISTORY_FILE

def load_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # Empty file safety
            if not content:
                return []

            return json.loads(content)

    except (json.JSONDecodeError, UnicodeDecodeError, IOError):
        # If file is corrupted, reset safely
        return []

def save_message(role, content):
    history = load_history()
    history.append({
        "role": role,
        "content": content
    })

    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def get_relevant_history(question):
    history = load_history()
    relevant = []

    q_words = set(question.lower().split())

    for msg in history[-15:]:
        try:
            msg_words = set(msg["content"].lower().split())
            if q_words.intersection(msg_words):
                relevant.append(msg["content"])
        except Exception:
            continue

    return "\n".join(relevant)
