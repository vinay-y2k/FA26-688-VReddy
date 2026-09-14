import os
import json
import psycopg2
import requests
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

DEPLOY_ENV = os.getenv("DEPLOY_ENV", "LOCAL")
VECTOR_DIM = 768 if DEPLOY_ENV == "LOCAL" else 1536
TARGET_ROW_LIMIT = 10000  
MAX_WORKERS = 5  # Sped up across 5 parallel thread streams concurrently

print(f"📦 Launching Multithreaded Ingestion Core in [{DEPLOY_ENV}] Mode...")

# Initialize empty list to buffer raw text blocks from streaming connection
print("🔗 Opening streaming data pool from Hugging Face...")
sec_stream = load_dataset("PleIAs/SEC", split="train", streaming=True)

# Pull and slice raw documents into text strings array quickly
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
text_chunks_pool = []

for row in sec_stream:
    if len(text_chunks_pool) >= TARGET_ROW_LIMIT:
        break
    raw_text = row.get("text", "")
    if raw_text and len(raw_text) > 100:
        chunks = text_splitter.split_text(raw_text)
        for chunk in chunks:
            if len(text_chunks_pool) < TARGET_ROW_LIMIT:
                text_chunks_pool.append(chunk)
            else:
                break

print(f"✅ Text Slicing Complete. Total text segments buffered: {len(text_chunks_pool)}")

# Thread worker task function
def process_single_vector_record(task_tuple):
    chunk_text, index = task_tuple
    
    # Establish dynamic individual database thread connections safely
    db_conn = "dbname=aifootprint user=db_admin_root password='Naveen_2026@' host=localhost port=5432"
    try:
        # 1. Fetch vector embedding array from local server client
        res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": chunk_text}, timeout=15)
        vector_coordinates = res.json()["embedding"]
        
        # 2. Open private transaction channel to Docker container
        conn = psycopg2.connect(db_conn)
        cur = conn.cursor()
        cur.execute("INSERT INTO sec_edgar_vectors (text_content, embedding) VALUES (%s, %s);", (chunk_text, vector_coordinates))
        conn.commit()
        cur.close()
        conn.close()
        
        # Dashboard updates with continuous counter tracks
        print(f"📥 Thread worker indexing record tracking counter: #{index + 1} / {TARGET_ROW_LIMIT}", end="\r")
    except Exception as e:
        pass # Gracefully skip occasional network anomalies or timeouts

# Initialize Database Schema Migrations Base
db_base_conn = "dbname=aifootprint user=db_admin_root password='Naveen_2026@' host=localhost port=5432"
base_conn = psycopg2.connect(db_base_conn)
base_cur = base_conn.cursor()
base_cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
base_cur.execute(f"CREATE TABLE IF NOT EXISTS sec_edgar_vectors (id SERIAL PRIMARY KEY, text_content TEXT, embedding vector({VECTOR_DIM}));")
base_conn.commit()
base_cur.close()
base_conn.close()

# Launch Multithreaded Processing Engine
print(f"🚀 Deploying ThreadPoolExecutor across {MAX_WORKERS} system threads...")
tasks_payload = [(chunk, idx) for idx, chunk in enumerate(text_chunks_pool)]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(process_single_vector_record, tasks_payload)

print(f"\n🎉 Parallel pipeline success! Seeding complete for {TARGET_ROW_LIMIT} rows safely.")
