import streamlit as st
import requests
import os
import tempfile
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ---------------- CONFIG ----------------
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3"

st.set_page_config(
    page_title="Personal Study Assistant AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- EMBEDDINGS SETUP ----------------
embedding_function = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="study_memory",
    embedding_function=embedding_function
)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "mode" not in st.session_state:
    st.session_state.mode = "Normal"

# ---------------- FUNCTIONS ----------------
def call_ollama(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False
    }
    response = requests.post(OLLAMA_CHAT_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]

def save_to_memory(text):
    collection.add(
        documents=[text],
        ids=[str(len(collection.get()["ids"]) + 1)]
    )

def retrieve_memory(query):
    results = collection.query(query_texts=[query], n_results=3)
    if results["documents"]:
        return "\n".join(results["documents"][0])
    return ""

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ---------------- UI ----------------
st.title("🧠 Personal Study Assistant AI")
st.caption("Python • SQL • AI/ML • Interview Prep (100% FREE, Offline)")

# Mode selection
st.session_state.mode = st.radio(
    "Select Mode",
    ["Normal", "Interview Mock"],
    horizontal=True
)

# PDF Upload
st.subheader("📄 Upload Notes / PDF")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    save_to_memory(pdf_text)
    st.success("✅ PDF content saved to memory")

# Question input
st.subheader("💬 Ask a Question")
question = st.text_area("Your Question")

if st.button("Ask Assistant"):
    if not question.strip():
        st.warning("Please enter a question")
    else:
        memory_context = retrieve_memory(question)

        if st.session_state.mode == "Interview Mock":
            system_prompt = (
                "You are an interviewer. Ask one question at a time. "
                "After the user's answer, give short feedback and ask the next question."
            )
        else:
            system_prompt = "You are a helpful study assistant."

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        if memory_context:
            messages.append({
                "role": "system",
                "content": f"Relevant notes:\n{memory_context}"
            })

        # Chat history
        for chat in st.session_state.chat_history:
            messages.append(chat)

        messages.append({"role": "user", "content": question})

        try:
            answer = call_ollama(messages)

            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            save_to_memory(question + "\n" + answer)

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- CHAT DISPLAY ----------------
st.subheader("🧾 Chat History")

for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**🧑 You:** {chat['content']}")
    else:
        st.markdown(f"**🤖 Assistant:** {chat['content']}")

# Clear chat
if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared")
