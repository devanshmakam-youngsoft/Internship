import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from google import genai
import json


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


# Create HF client
LLM = genai.Client(api_key=api_key)

# Pydantic model for structured output
class QueryVariations(BaseModel):
    queries: List[str]
# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Google make money?"
print(f"Original Query: {original_query}\n")

# ──────────────────────────────────────────────────────────────────
# Step 1: Generate Multiple Query Variations
# ──────────────────────────────────────────────────────────────────

prompt = f"""
You are a system that ONLY outputs valid JSON.
Do NOT include explanations, markdown, or extra text.

Generate 3 different variations of this query that would help retrieve relevant documents.

Original query:
{original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles.

Output MUST be exactly in the following JSON format:

{{
  "queries": [
    "query variation 1",
    "query variation 2",
    "query variation 3"
  ]
}}
"""

raw_response = LLM.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)


response_text = raw_response.text.strip()

query_variations_obj = QueryVariations.model_validate_json(response_text)
query_variations = query_variations_obj.queries


print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────
# Step 2: Search with Each Query Variation & Store Results
# ──────────────────────────────────────────────────────────────────

retriever = db.as_retriever(search_kwargs={"k": 5})  # Get more docs for better RRF
all_retrieval_results = []  # Store all results for RRF

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)  # Store for RRF calculation
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")
    
    print("-" * 50)

print("\n" + "="*60)
print("Multi-Query Retrieval Complete!")


# all_retrieval_results = [
#     [Doc1, Doc2, Doc3, Doc4, Doc5],  ← Query 1 results
#     [Doc2, Doc1, Doc6, Doc7, Doc3],  ← Query 2 results  
#     [Doc8, Doc2, Doc9, Doc10, Doc11] ← Query 3 results
# ]