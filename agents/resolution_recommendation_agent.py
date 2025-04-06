
import chromadb
import os
from gen_model import run
from sentence_transformers import SentenceTransformer
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="ticket_resolutions")


embedder = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Generate embedding for text using Ollama"""
    # response = ollama.embeddings(model=OLLAMA_MODEL, prompt=text)
    # return response["embedding"]
    return embedder.encode(text).tolist()



def retrieve_similar_cases(query_text, top_k=3):
    """Find similar past tickets using ChromaDB"""
    query_embedding = embed_text(query_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    return [(doc["resolution"], score) for doc, score in zip(results["metadatas"][0], results["distances"][0])]


def generate_resolution(extracted_actions):
    """Generate a recommended resolution using retrieved cases & LLM"""
    similar_cases = retrieve_similar_cases(extracted_actions)

    context = "\n".join([f"Case: {case}\nSimilarity: {score}" for case, score in similar_cases])

    prompt = f"""
    You are a technical support resolution agent.

    A customer has reported the following issue:
    "{extracted_actions}"

    We have found similar past cases with their resolutions:

    {context}

    Based on these similar cases, generate a recommended resolution for the current issue.
    - Use technical language where appropriate.
    - Keep the answer clear and actionable.
    - Start your response with: Resolution:

    Only provide the resolution, not explanations about similarity or previous cases.
    """

    response = run(prompt)
    return response




