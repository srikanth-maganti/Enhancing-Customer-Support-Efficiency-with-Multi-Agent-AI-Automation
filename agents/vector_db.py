import pandas as pd
import chromadb
import os
from sentence_transformers import SentenceTransformer
# Load CSV file safely
file_path = os.path.abspath("../data/Historical_ticket_data.csv")
df = pd.read_csv(file_path)

# Strip column names of whitespace
df.columns = df.columns.str.strip()

# Initialize ChromaDB client
db_client = chromadb.PersistentClient(path="../chroma_db")
collection = db_client.get_or_create_collection(name="ticket_resolutions")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
# Load Ollama model


def embed_text(text):
    """Generate embedding for text using Ollama"""
    return embedder.encode(text).tolist()

# Required columns
required_cols = ["Issue Category", "Solution", "Team", "Resolution time", "Resolution Status"]

if all(col in df.columns for col in required_cols):
    for index, row in df.iterrows():

        ticket_text = row["Issue Category"]
        resolution = row["Solution"]
        team = row["Team"]
        resolution_time = row["Resolution time"]
        resolution_status = row["Resolution Status"]

        embedding = embed_text(ticket_text)

        # Store in ChromaDB
        collection.add(
            embeddings=[embedding],
            documents=[ticket_text],
            metadatas=[{
                "resolution": resolution,
                "team": team,
                "resolution_time": resolution_time,
                "resolution_status": resolution_status
            }],
            ids=[str(index)]
        )

    print("✅ All historical ticket data successfully added to ChromaDB!")
else:
    print("❌ Error: Missing one or more required columns in the CSV.")
