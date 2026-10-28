import os
import time
import csv
import psycopg2
import requests
import numpy as np
from pathlib import Path
from dotenv import dotenv_values

# Get the path to the directory where this script lives
script_dir = Path(__file__).resolve().parent
env_path = script_dir / ".." / ".env"

config = dotenv_values(env_path)

CSV_OUTPUT_FILE = script_dir/ ".." / "csv_image_files" / "inference_timing_benchmarks.csv"

# 15 Test prompts explicitly curated to evaluate your semantic boundaries
TEST_SUITE = [
    # Path 1: Financial RAG Prompts
    {"query": "What is the primary manufacturing scope and business overview of Imo Industries?", "expected": "Path 1 (RAG)"},
    {"query": "Show me the net corporate revenues and operational updates logged in 1993.", "expected": "Path 1 (RAG)"},
    {"query": "What are the material operational risks and segment liabilities listed?", "expected": "Path 1 (RAG)"},
    {"query": "Examine the balance sheets for cash positions and quarterly earnings.", "expected": "Path 1 (RAG)"},
    {"query": "Detail the business segments and core power products manufacturing details.", "expected": "Path 1 (RAG)"},
    
    # Path 2: Database Administration Prompts
    {"query": "Show the current table size metrics and database footprint weight.", "expected": "Path 2 (Admin)"},
    {"query": "What is the physical storage size and block footprint of sec_table_1?", "expected": "Path 2 (Admin)"},
    {"query": "Examine database metrics, schema indexes, and vacuum row counts.", "expected": "Path 2 (Admin)"},
    {"query": "Provide the storage size metadata footprint for the vector tables.", "expected": "Path 2 (Admin)"},
    {"query": "Show database metrics and physical index storage sizes.", "expected": "Path 2 (Admin)"},
    
    # Path 3: Fallback General Conversational Prompts
    {"query": "Explain the difference between a list and a tuple in Python.", "expected": "Path 3 (Fallback)"},
    {"query": "Write a clean recursive function in Python to calculate a factorial.", "expected": "Path 3 (Fallback)"},
    {"query": "What are the primary structural differences between a B-Tree and an HNSW graph?", "expected": "Path 3 (Fallback)"},
    {"query": "How many states are there in the United States?", "expected": "Path 3 (Fallback)"},
    {"query": "Draft a professional email script notifying an advisor about a project status change.", "expected": "Path 3 (Fallback)"}
]

INTENT_ANCHORS = {
    "financial": "Corporate financials, company revenue, net income statement, quarterly earnings, or business operational updates.",
    "admin": "Database tables size, metrics, physical storage footprints, schema indexes, or vacuum row counts."
}

def get_embedding(text_prompt):
    url = "http://localhost:11434/api/embeddings"
    try:
        res = requests.post(url, json={"model": "nomic-embed-text", "prompt": text_prompt}, timeout=5)
        return np.array(res.json().get("embedding", []), dtype=np.float32)
    except Exception:
        return np.zeros(768, dtype=np.float32)

def compute_true_cosine_distance(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0: return 1.0
    return max(0.0, 1.0 - (dot_product / (norm_a * norm_b)))

def query_llm_with_timing(prompt_string):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:3b",
        "prompt": prompt_string,
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 4096}
    }
    start_time = time.perf_counter()
    response = requests.post(url, json=payload, timeout=45)
    end_time = time.perf_counter()
    
    duration = (end_time - start_time) * 1000  # Convert to milliseconds
    response_json = response.json()
    eval_count = response_json.get("eval_count", 0)
    tokens_per_second = eval_count / (duration / 1000) if duration > 0 else 0.0
    return round(duration, 2), round(tokens_per_second, 2)

def execute_benchmarks():
    print("Initializing Automated Router Performance Diagnostics...")
    
    financial_vector = get_embedding(INTENT_ANCHORS["financial"])
    admin_vector = get_embedding(INTENT_ANCHORS["admin"])
    
    benchmark_records = []
    
    for idx, item in enumerate(TEST_SUITE):
        user_query = item["query"]
        query_vector = get_embedding(user_query)
        
        dist_financial = compute_true_cosine_distance(query_vector, financial_vector)
        dist_admin = compute_true_cosine_distance(query_vector, admin_vector)
        delta_margin = abs(dist_financial - dist_admin)
        
        # Enforce confidence delta gating rules
        is_ambiguous = delta_margin < 0.05
        
        route_chosen = "Path 3: Fallback"
        latency_ms = 0.0
        tokens_per_sec = 0.0
        
        if is_ambiguous:
            route_chosen = "Path 3: Fallback (Ambiguous Block)"
            latency_ms, tokens_per_sec = query_llm_with_timing(user_query)
        elif dist_financial <= 0.42:
            route_chosen = "Path 1: Financial RAG"
            
            # Simulated RAG Table Lookup
            # db_conn = "dbname=aifootprint user=db_admin_root password=your_secure_password host=localhost port=5432"
            db_conn = f"dbname={config['DB_NAME']} user={config['DB_USER']} password={config['DB_PASSWORD']} host=localhost port=5432"
            conn = psycopg2.connect(db_conn)
            cur = conn.cursor()
            cur.execute("SELECT text_content FROM sec30_table_1 ORDER BY embedding <=> %s::vector LIMIT 1;", (query_vector.tolist(),))
            context_block = cur.fetchone()
            cur.close()
            conn.close()
            
            prompt = f"Context: {context_block}\nQuestion: {user_query}\nAnswer factually using context:"
            latency_ms, tokens_per_sec = query_llm_with_timing(prompt)
        elif dist_admin <= 0.45:
            route_chosen = "Path 2: NL-DBA Bypass"
            
            # Metadata Bypass Measurement (Zero LLM overhead)
            start_meta = time.perf_counter()
            # db_conn = "dbname=aifootprint user=db_admin_root password=your_secure_password host=localhost port=5432"
            db_conn = f"dbname={config['DB_NAME']} user={config['DB_USER']} password={config['DB_PASSWORD']} host=localhost port=5432"
            conn = psycopg2.connect(db_conn)
            cur = conn.cursor()
            cur.execute("SELECT pg_size_pretty(pg_table_size('sec30_table_1'));")
            cur.fetchone()
            cur.close()
            conn.close()
            end_meta = time.perf_counter()
            
            latency_ms = round((end_meta - start_meta) * 1000, 2)
            tokens_per_sec = 0.0  # Bypassed model generation entirely
        else:
            latency_ms, tokens_per_sec = query_llm_with_timing(user_query)
            
        print(f" Prompt {idx+1:02d} | Route: {route_chosen:<28} | Latency: {latency_ms:8.2f}ms | Throughput: {tokens_per_sec:5.1f} t/s")
        
        benchmark_records.append({
            "prompt_id": idx + 1,
            "query_text": user_query,
            "dist_financial": round(dist_financial, 3),
            "dist_admin": round(dist_admin, 3),
            "delta_margin": round(delta_margin, 3),
            "route_chosen": route_chosen,
            "latency_ms": latency_ms,
            "tokens_per_second": tokens_per_sec
        })
        
    # Write metrics to target data log spreadsheet
    with open(CSV_OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "query_text", "dist_financial", "dist_admin", "delta_margin", "route_chosen", "latency_ms", "tokens_per_second"])
        writer.writeheader()
        writer.writerows(benchmark_records)
        
    print(f"\n🎉 Performance logs compiled successfully inside '{CSV_OUTPUT_FILE}'!")

if __name__ == "__main__":
    execute_benchmarks()
