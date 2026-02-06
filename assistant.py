import requests
from memory import search_memory, save_to_memory

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_assistant(question):
    context = search_memory(question)

    prompt = f"""
You are a helpful study assistant.

Context:
{context}

Question:
{question}
"""

    payload = {
        "model": "gemma3:1b",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    answer = response.json()["response"]
    save_to_memory(question + " " + answer)

    return answer
