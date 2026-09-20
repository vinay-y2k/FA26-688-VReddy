import os
import json
import time
import csv
import psycopg2
import requests
from pathlib import Path
from dotenv import dotenv_values

# Get the path to the directory where this script lives
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".env"

config = dotenv_values(env_path)

# 1. PLATFORM CONFIGURATION SETTINGS
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "LOCAL")
CSV_OUTPUT_FILE = "vector_search_benchmarks.csv"

# Standardized financial queries - 1 strictly mapped to 1 isolated table structure
BENCHMARK_QUERIES = [
    "What is the primary business overview and manufacturing scope?",
    "Show me the net operational revenue metrics and year over year growth.",
    "What are the material market risks and infrastructure liabilities listed?",
    "Examine the balance sheets for cash equivalents and current positions.",
    "What long term database storage agreements are signed with cloud providers?",
    "Detail the capital expenditures required for hardware engine manufacturing.",
    "Are there any credit facilities or active liquidity risk adjustments?",
    "Show text references discussing corporate operations margins stability.",
    "What are the primary segments and divisions within the core business?",
    "Detail long term lease commitments for factories and infrastructure hubs.",
    "What compliance frameworks or environmental rules affect operations?",
    "Show research and development investment trends across product lines.",
    "Are there any material litigation histories or active legal exposures?",
    "What is the exact fiscal year reporting timeline for this statement?",
    "Detail cash allocations spent on database container computing nodes.",
    "What inflationary pressures represent material risks to operational capacity?",
    "Show notes discussing asset impairment values or inventory adjustments.",
    "What debt instruments or corporate notes are listed under liabilities?",
    "Examine the statement of cash flows for operating activities allocations.",
    "What customer concentration metrics represent key structural dependencies?",
    "Detail executive compensation structures or stock performance metrics.",
    "What international trade variables impact manufacturing supply chains?",
    "Show text blocks discussing pension liabilities or employee obligations.",
    "What restructuring plans or factory optimization initiatives are logged?",
    "Detail short term cash positions compared against total interest charges.",
    "What cybersecurity governance measures are used to protect financial ledgers?",
    "Are there any segment revenue distributions recorded for tech sectors?",
    "Examine tax provisions, deferred tax assets, and federal calculations.",
    "What raw materials market volatility impacts production line margins?",
    "Show the final audit report sign off and accountant verification notes."
]

# def get_query_vector(text_prompt):
#     """Generates 768-dimensional coordinate spaces using local GPU layers"""
#     res = requests.post("http://localhost:11434/api/embeddings", json={"model": "nomic-embed-text", "prompt": text_prompt})
#     return res.json()["embedding"]

def get_query_vector(text_prompt):
    """Generates 768-dimensional coordinate spaces using local GPU layers with timeout safety"""
    url = "http://localhost:11434/api/embeddings"
    try:
        # Enforcing a strict 5-second connection timeout safeguard
        res = requests.post(
            url, 
            json={"model": "nomic-embed-text", "prompt": text_prompt}, 
            timeout=5
        )
        return res.json()["embedding"]
    except requests.exceptions.Timeout:
        raise SystemExit("System Error: Ollama connection timed out. Verify your 'ollama serve' terminal is active.")
    except requests.exceptions.ConnectionError:
        raise SystemExit("System Error: Connection refused on port 11434. Please restart the Ollama service.")


def execute_30_table_benchmark():
    # db_conn = "dbname=aifootprint user=db_admin_root password=your_secure_password host=localhost port=5432"
    db_conn = f"dbname={config['DB_NAME']} user={config['DB_USER']} password={config['DB_PASSWORD']} host=localhost port=5432"
    print(f"Initializing Production-Scale Benchmarking Harness in [{DEPLOY_ENV}] Mode...")
    print(f"Distributing 30 distinct inquiries sequentially across 30 separate 10,000-row tables...\n")
    
    try:
        conn = psycopg2.connect(db_conn)
        cur = conn.cursor()
        
        benchmark_records = []
        
        for idx, question in enumerate(BENCHMARK_QUERIES):
            query_vector = get_query_vector(question)
            
            # Map query 0 to table 1, query 1 to table 2, up to query 29 to table 30
            target_table = f"sec30_table_{idx + 1}"
            
            # --- RUN 1: SCAN FORCING UNINDEXED SEQUENTIAL LOOKUP (O(N)) ---
            cur.execute("SET enable_indexscan = OFF; SET enable_bitmapscan = OFF;")
            conn.commit()
            
            start_unindexed = time.perf_counter()
            cur.execute(
                f"SELECT text_content FROM {target_table} ORDER BY embedding <=> %s::vector LIMIT 1;",
                (query_vector,)
            )
            cur.fetchone()
            end_unindexed = time.perf_counter()
            latency_unindexed_ms = (end_unindexed - start_unindexed) * 1000
            
            # --- RUN 2: SCAN UTILIZING FINE-TUNED HNSW GRAPH INDEX (O(log N)) ---
            cur.execute("SET enable_indexscan = ON; SET enable_bitmapscan = ON;")
            conn.commit()
            
            start_indexed = time.perf_counter()
            cur.execute(
                f"SELECT text_content FROM {target_table} ORDER BY embedding <=> %s::vector LIMIT 1;",
                (query_vector,)
            )
            cur.fetchone()
            end_indexed = time.perf_counter()
            latency_indexed_ms = (end_indexed - start_indexed) * 1000
            
            speed_gain_multiplier = latency_unindexed_ms / latency_indexed_ms
            
            print(f"Query #{idx+1:02d} -> [{target_table}] | Unindexed: {latency_unindexed_ms:6.2f}ms | HNSW Graph: {latency_indexed_ms:5.2f}ms | Speedup: {speed_gain_multiplier:5.1f}x")
            
            benchmark_records.append({
                "query_id": idx + 1,
                "user_query": question,
                "latency_unindexed_ms": round(latency_unindexed_ms, 2),
                "latency_indexed_ms": round(latency_indexed_ms, 2),
                "speed_multiplier_gain": round(speed_gain_multiplier, 1)
            })
            
            # Clear internal database query caches between tracking changes
            cur.execute("DISCARD PLANS; DISCARD TEMPORARY;")
            conn.commit()
            
        cur.close()
        conn.close()
        
        # Write performance array straight to target data log file
        with open(CSV_OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "user_query", "latency_unindexed_ms", "latency_indexed_ms", "speed_multiplier_gain"])
            writer.writeheader()
            writer.writerows(benchmark_records)
            
        print(f"\n Benchmarking Complete! Clean performance metrics compiled inside '{CSV_OUTPUT_FILE}'.")
        
    except Exception as e:
        print(f"\n Operational Performance Check Failed: {e}")

if __name__ == "__main__":
    execute_30_table_benchmark()
