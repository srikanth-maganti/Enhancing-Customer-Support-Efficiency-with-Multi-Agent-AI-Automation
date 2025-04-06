import autogen

import chromadb
from sentence_transformers import SentenceTransformer
from gen_model import run
# Initialize ChromaDB client and collection
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="ticket_resolutions")

embedder = SentenceTransformer('all-MiniLM-L6-v2')


import pandas as pd
import os

EXCEL_PATH = "./routed_tickets.xlsx"  # File to store routed tickets

def store_task_routing(ticket_id, extracted_actions, recommended_resolution, assigned_team):
    """Store the task details into an Excel sheet with unique ticket ID"""

    # Create a DataFrame for the current ticket
    new_data = pd.DataFrame([{
        "Ticket ID": ticket_id,
        "Extracted Actions": extracted_actions,
        "Recommended Resolution": recommended_resolution,
        "Assigned Team": assigned_team
    }])

    # Check if the file already exists
    if os.path.exists(EXCEL_PATH):
        # Read existing data
        existing_data = pd.read_excel(EXCEL_PATH)

        # Check for duplicate Ticket ID before adding
        if ticket_id not in existing_data["Ticket ID"].values:
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            updated_data.to_excel(EXCEL_PATH, index=False)
            print(f"✅ Ticket {ticket_id} stored in Excel.")
        else:
            print(f"⚠️ Ticket {ticket_id} already exists in Excel. Skipping.")
    else:
        # File doesn't exist, create a new one
        new_data.to_excel(EXCEL_PATH, index=False)
        print(f"✅ New Excel file created. Ticket {ticket_id} stored.")

def embed_text(text):
    """Generate embedding for text using Ollama"""
    return embedder.encode(text).tolist()

def retrieve_similar_tasks(query_text, top_k=3):
    query_embedding = embed_text(query_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results if results else {}


def predict_routing_team(extracted_actions, recommended_resolution):
    """Predict the correct team using Ollama LLM"""
    similar_results = retrieve_similar_tasks(extracted_actions, top_k=3)

    context_lines = []
    time=0
    if "documents" in similar_results and "metadatas" in similar_results:
        for doc, meta in zip(similar_results["documents"][0], similar_results["metadatas"][0]):
            time+=meta["resolution_time"]

            case_summary = f"Issue: {doc}\nResolution: {meta['resolution']}\nTeam: {meta['team']}"
            context_lines.append(case_summary)

    if len(context_lines)==0:
        est_time=-1
    else:
        est_time=round(time/len(context_lines))

    context = "\n---\n".join(context_lines)

    prompt = f"""
You are an expert support ticket router in a customer service system.

Your job is to assign incoming issues to the correct team based on:
- The action extracted from the user's message
- The recommended resolution
- Previously handled similar cases

---

Extracted Action: "{extracted_actions}"
Recommended Resolution: "{recommended_resolution}"

Previously Routed Cases:
{context}

---

From the options below:
- Billing
- Network
- Software
- Technical

Based on the above information, respond with the **best team name only**, without explanation.
"""

    response =run(prompt)
    return (response.strip(),est_time)


def handle_message(ticket_id, extracted_actions, recommended_resolution):
    """Route ticket to the correct team and store it"""

    assigned_team,est_time = predict_routing_team(extracted_actions, recommended_resolution)

    store_task_routing(ticket_id, extracted_actions, recommended_resolution, assigned_team)
    return (assigned_team,est_time)


