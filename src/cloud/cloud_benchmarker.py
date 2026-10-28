import os
import time
import csv
import json
import boto3
import psycopg2
from pathlib import Path


# Get the path to the directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
# print(SCRIPT_DIR)


DB_PASSWORD = os.environ.get("TF_VAR_db_password")
DB_HOST = "capstone-aurora-serverless-vector-db.cluster-cmvycoausa3m.us-east-1.rds.amazonaws.com"
DB_CONN_STR = f"dbname=aws_aifootprint user=aws_db_admin_root password={DB_PASSWORD} host={DB_HOST} port=5432"
CSV_OUTPUT_FILE = SCRIPT_DIR/ ".." / "csv_image_files" / "cloud_search_benchmarks.csv"

# Re-uses the exact same 30 query lines from your local benchmarking harness
from benchmarking_runner_final import BENCHMARK_QUERIES 

def get_titan_embedding(client, text):
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = client.invoke_model(modelId="amazon.titan-embed-text-v2:0", contentType="application/json", accept="application/json", body=body)
    return json.loads(response.get("body").read()).get("embedding")

def execute_cloud_benchmarks():
    print("Launching 30-Table Cloud Vector Performance Evaluations...")
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    conn = psycopg2.connect(DB_CONN_STR)
    cur = conn.cursor()
    
    benchmark_records = []
    
    for idx, question in enumerate(BENCHMARK_QUERIES):
        query_vector = get_titan_embedding(bedrock_client, question)
        target_table = f"aws_sec_table_{idx + 1}"
        
        # Run 1: Unindexed Sequential Scan (Forcing O(N))
        cur.execute("SET enable_indexscan = OFF; SET enable_bitmapscan = OFF;")
        conn.commit()
        start_unindexed = time.perf_counter()
        cur.execute(f"SELECT text_content FROM {target_table} ORDER BY embedding <=> %s::vector LIMIT 1;", (query_vector,))
        cur.fetchone()
        latency_unindexed = (time.perf_counter() - start_unindexed) * 1000
        
        # Run 2: Fine-Tuned HNSW Graph Scan (Forcing O(log N))
        cur.execute("SET enable_indexscan = ON; SET enable_bitmapscan = ON;")
        conn.commit()
        start_indexed = time.perf_counter()
        cur.execute(f"SELECT text_content FROM {target_table} ORDER BY embedding <=> %s::vector LIMIT 1;", (query_vector,))
        cur.fetchone()
        latency_indexed = (time.perf_counter() - start_indexed) * 1000
        
        print(f"{target_table} | Unindexed: {latency_unindexed:6.2f}ms | HNSW: {latency_indexed:5.2f}ms | Gain: {latency_unindexed/latency_indexed:5.1f}x")
        
        benchmark_records.append({
            "query_id": idx + 1,
            "latency_unindexed_ms": round(latency_unindexed, 2),
            "latency_indexed_ms": round(latency_indexed, 2),
            "speed_multiplier_gain": round(latency_unindexed / latency_indexed, 1)
        })
        cur.execute("DISCARD PLANS; DISCARD TEMPORARY;")
        conn.commit()

    with open(CSV_OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "latency_unindexed_ms", "latency_indexed_ms", "speed_multiplier_gain"])
        writer.writeheader()
        writer.writerows(benchmark_records)
        
    cur.close()
    conn.close()
    print(f" Complete! Logs compiled inside '{CSV_OUTPUT_FILE}'.")

if __name__ == "__main__":
    execute_cloud_benchmarks()
