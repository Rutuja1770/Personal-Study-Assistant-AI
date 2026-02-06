from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="./db",
    embedding_function=embedding
)

def save_to_memory(text: str):
    db.add_texts([text])
    db.persist()

def search_memory(query: str):
    results = db.similarity_search(query, k=3)
    return [r.page_content for r in results]
from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("study_memory")

model = SentenceTransformer("all-MiniLM-L6-v2")

def save_to_memory(text):
    embedding = model.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(len(collection.get()["ids"]) + 1)]
    )

def search_memory(query):
    embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )
    return " ".join(results["documents"][0]) if results["documents"] else ""
