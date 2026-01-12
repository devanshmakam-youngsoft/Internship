from llm import call_llm
from utils import extract_json

def plan_actions(question, memory_context):
    prompt = f"""
You are an MCP planner.

Return ONLY JSON:
{{
  "plan": ["memory", "web", "llm"],
  "reason": "short reason"
}}

Question:
{question}

Memory:
{memory_context}
"""
    raw = call_llm(prompt)
    return extract_json(
        raw,
        fallback={"plan": ["llm"], "reason": "fallback"}
    )

def generate_answer(question, context):
    prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{question}

Return ONLY JSON:
{{
  "answer": "",
  "confidence": 0-100
}}
"""
    raw = call_llm(prompt)
    return extract_json(
        raw,
        fallback={
            "answer": context[:500] if context else "Unable to answer.",
            "confidence": 40
        }
    )
