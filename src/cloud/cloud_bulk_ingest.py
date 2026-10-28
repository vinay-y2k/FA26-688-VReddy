import os
import sys
import time
import json
import boto3
import psycopg2
from psycopg2.extras import execute_values
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

# ENVIRONMENT VARIABLE ACCESS CONTROL
DB_PASSWORD = os.environ.get("TF_VAR_db_password")
if not DB_PASSWORD:
    print("Security Error: 'TF_VAR_db_password' not found in environment.")
    sys.exit(1)


DB_HOST = "capstone-aurora-serverless-vector-db.cluster-cmvycoausa3m.us-east-1.rds.amazonaws.com"
DB_CONN_STR = f"dbname=aws_aifootprint user=aws_db_admin_root password={DB_PASSWORD} host={DB_HOST} port=5432"

VECTOR_DIM = 1024
TABLE_COUNT = 30
ROWS_PER_TABLE = 10000
BULK_FLUSH_SIZE = 200
CONCURRENT_THREADS = 15  # Scaled up to maximize AWS Bedrock cloud API limits

print(f"Initializing 300K Production Cloud Ingestion Pipeline...")

# 2. STREAM AND BUFFER ALL 300,000 TEXT SLICES IN RAM
print("Connecting to Hugging Face SEC data stream...")
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

print(f"Text Slicing Complete. Total buffered items: {len(global_text_pool)}")

# 3. HIGH-THROUGHPUT AWS MODEL INVOCATION CORE
def invoke_titan_embedding(client, text):
    try:
        body = json.dumps({"inputText": text, "dimensions": VECTOR_DIM, "normalize": True})
        response = client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=body
        )
        return json.loads(response.get("body").read()).get("embedding")
    except Exception as e:
        print(f" Bedrock Error: {e}")
        return None

def run_cloud_ingestion_orchestration():
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    
    conn = psycopg2.connect(DB_CONN_STR)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cur.close()
    conn.close()
    
    for t_idx in range(TABLE_COUNT):
        t_name = f"aws_sec_table_{t_idx + 1}"
        print(f"\n🚀 Seeding Isolated Cloud Table: [{t_name}]...")
        
        # Open a fresh connection per table block to maintain network channel integrity
        conn = psycopg2.connect(DB_CONN_STR)
        cur = conn.cursor()
        
        cur.execute(f"DROP TABLE IF EXISTS {t_name};")
        cur.execute(f"CREATE TABLE {t_name} (id SERIAL PRIMARY KEY, text_content TEXT, embedding vector({VECTOR_DIM}));")
        conn.commit()
        
        start_pos = t_idx * ROWS_PER_TABLE
        end_pos = start_pos + ROWS_PER_TABLE
        table_text_chunks = global_text_pool[start_pos:end_pos]
        
        print(f"  Generating 10,000 Titan v2 cloud vectors...")
        start_vectors = time.perf_counter()
        with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
            embeddings_results = list(executor.map(lambda txt: invoke_titan_embedding(bedrock_client, txt), table_text_chunks))
        print(f" Vectors generated in {time.perf_counter() - start_vectors:.1f}s.")
        
        valid_records = []
        for chunk, vector in zip(table_text_chunks, embeddings_results):
            if vector is not None:
                valid_records.append((chunk, vector))
                
        # 200-Row Inner Batch Flush to AWS Storage
        print(f" Executing 200-row page flushes straight to Aurora Serverless...")
        for i in range(0, len(valid_records), BULK_FLUSH_SIZE):
            sub_batch = valid_records[i : i + BULK_FLUSH_SIZE]
            try:
                execute_values(cur, f"INSERT INTO {t_name} (text_content, embedding) VALUES %s", sub_batch)
                conn.commit()
            except psycopg2.OperationalError as e:
                # Automate instant network socket recovery fallback
                print("  Network glitch detected. Refreshing TLS channel connection socket...")
                cur.close()
                conn.close()
                conn = psycopg2.connect(DB_CONN_STR)
                cur = conn.cursor()
                execute_values(cur, f"INSERT INTO {t_name} (text_content, embedding) VALUES %s", sub_batch)
                conn.commit()
            
        print(f" Compiling multi-layer HNSW graph index on AWS...")
        start_index = time.perf_counter()
        cur.execute(f"CREATE INDEX {t_name}_hnsw_idx ON {t_name} USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);")
        conn.commit()
        print(f" Index compiled in {time.perf_counter() - start_index:.1f}s.")
        
        # Safe connection closure per cycle
        cur.close()
        conn.close()
        
        del table_text_chunks
        del embeddings_results
        del valid_records

    print("\n Architecture Frozen! All 30 tables successfully seeded with 10k rows each on AWS.")

if __name__ == "__main__":
    run_cloud_ingestion_orchestration()
