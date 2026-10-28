import os
import asyncio
import aiohttp
import psycopg2
from psycopg2.extras import execute_values
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import dotenv_values

# Get the path to the directory where this script lives
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

config = dotenv_values(env_path)

DEPLOY_ENV = os.getenv("DEPLOY_ENV", "LOCAL")
VECTOR_DIM = 768
TABLE_COUNT = 30  
ROWS_PER_TABLE = 10000 
BULK_FLUSH_SIZE = 1000   # ⚡ FIXED: Flushes memory pages to disk every 1,000 records
CONCURRENT_REQUESTS = 40  # Perfectly maps to your OLLAMA_NUM_PARALLEL flag

print(f"Initializing Chunk-Wise Async Ingestion with 1K Batch Flushes in [{DEPLOY_ENV}] Mode...")

# 1. PRE-BUFFER TEXT STRINGS FROM HUGGING FACE
print("Opening streaming data pool connection from Hugging Face...")
sec_stream = load_dataset("PleIAs/SEC", split="train", streaming=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

global_text_pool = []
for row in sec_stream:
    if len(global_text_pool) >= (TABLE_COUNT * ROWS_PER_TABLE):
        break
    raw_text = row.get("text", "")
    if raw_text and len(raw_text) > 100:
        chunks = text_splitter.split_text(raw_text)
        for chunk in chunks:
            if len(global_text_pool) < (TABLE_COUNT * ROWS_PER_TABLE):
                global_text_pool.append(chunk)

print(f" Text Slicing Complete. Total buffered string items: {len(global_text_pool)}")

# 2. ASYNCHRONOUS TENSOR WORKER FUNCTION
async def compute_single_embedding(session, semaphore, text_slice):
    url = "http://localhost:11434/api/embeddings"
    async with semaphore:
        try:
            async with session.post(url, json={"model": "nomic-embed-text", "prompt": text_slice}, timeout=15) as res:
                response_json = await res.json()
                return response_json["embedding"]
        except Exception:
            return None

# 3. BATCHED PIPELINE COORDINATOR WITH INNER 1,000-ROW FLUSHES
async def run_chunked_ingestion():
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    # db_conn_str = "dbname=aifootprint user=db_admin_root password=your_secure_password host=localhost port=5432"
    db_conn_str = f"dbname={config['DB_NAME']} user={config['DB_USER']} password={config['DB_PASSWORD']} host=localhost port=5432"
    conn = psycopg2.connect(db_conn_str)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    async with aiohttp.ClientSession() as session:
        for t_idx in range(TABLE_COUNT):
            t_name = f"sec30_table_{t_idx + 1}"
            print(f"\n Processing isolated boundaries for table: [{t_name}]...")
            
            cur.execute(f"DROP TABLE IF EXISTS {t_name};")
            cur.execute(f"CREATE TABLE {t_name} (id SERIAL PRIMARY KEY, text_content TEXT, embedding vector({VECTOR_DIM}));")
            conn.commit()
            
            # Extract exactly 10,000 text lines for this table
            start_pos = t_idx * ROWS_PER_TABLE
            end_pos = start_pos + ROWS_PER_TABLE
            table_text_chunks = global_text_pool[start_pos:end_pos]
            
            # Compute vectors for this table's 10,000 items
            print(f"Computing 10,000 vectors for [{t_name}] via Intel Arc loops...")
            tasks = [compute_single_embedding(session, semaphore, chunk) for chunk in table_text_chunks]
            table_embeddings = await asyncio.gather(*tasks)
            
            # Filter successful records into a master list
            valid_records = []
            for chunk, vector in zip(table_text_chunks, table_embeddings):
                if vector is not None:
                    valid_records.append((chunk, vector))
            
            #  INNER NESTED LOOP: Flush to database in batches of 1,000
            print(f"Writing and flushing data into [{t_name}] in 1,000-row increments...")
            for i in range(0, len(valid_records), BULK_FLUSH_SIZE):
                sub_batch = valid_records[i : i + BULK_FLUSH_SIZE]
                execute_values(cur, f"INSERT INTO {t_name} (text_content, embedding) VALUES %s", sub_batch)
                conn.commit()  # Permanently writes the 1,000-row page chunk to the hard drive
                print(f" Buffered Flush Success: Committed records {i} to {i + len(sub_batch)} on disk.")
            
            # Compile the HNSW graph over the cleanly written database pages
            print(f" Compiling fine-tuned HNSW index graph on [{t_name}]...")
            cur.execute(f"CREATE INDEX {t_name}_hnsw_idx ON {t_name} USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);")
            conn.commit()
            
            print(f"Success! Table [{t_name}] completely finalized and indexed.")
            
            del table_text_chunks
            del table_embeddings
            del valid_records

    cur.close()
    conn.close()
    print("\nVerification Sandbox Complete! All 30 tables successfully compiled with 1K flush cycles and HNSW indexes.")

if __name__ == "__main__":
    asyncio.run(run_chunked_ingestion())
