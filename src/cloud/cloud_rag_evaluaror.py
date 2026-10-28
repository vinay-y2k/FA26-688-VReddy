import os
import sys
import time
import json
import csv
import boto3
import psycopg2
import numpy as np
from pathlib import Path


# Get the path to the directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
# print(SCRIPT_DIR)

# ENFORCE ENVIRONMENT SECURITY GATEWAYS
DB_PASSWORD = os.environ.get("TF_VAR_db_password")
if not DB_PASSWORD:
    print("Security Error: 'TF_VAR_db_password' environment token not detected in memory.")
    sys.exit(1)

# [TODO: Ensure this matches your active, publicly accessible Terraform cluster string endpoint]
DB_HOST = "capstone-aurora-serverless-vector-db.cluster-cmvycoausa3m.us-east-1.rds.amazonaws.com"
DB_CONN_STR = f"dbname=aws_aifootprint user=aws_db_admin_root password={DB_PASSWORD} host={DB_HOST} port=5432 sslmode=require"
CSV_OUTPUT_FILE = SCRIPT_DIR/ ".." / "csv_image_files" /"cloud_rag_inference_benchmarks.csv"

# Import your standardized 30 query lines from your benchmarking workspace
from benchmarking_runner_final import BENCHMARK_QUERIES

def get_titan_embedding(client, text):
    """Generates an L2-normalized 1024-dim cloud vector space"""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0", 
        contentType="application/json", 
        accept="application/json", 
        body=body
    )
    return json.loads(response.get("body").read()).get("embedding")

def query_nova_micro(client, prompt_string):
    """Invokes Amazon Nova Micro and records precise generation token analytics"""
    url = "https://amazonaws.com"
    
    # Corrected parameter configuration block structure for Amazon Nova v1
    body = json.dumps({
        "inferenceConfig": {
            "maxTokens": 400,  # Changed from maxNewTokens to maxTokens to pass AWS validation gates
            "temperature": 0.0,
            "topP": 0.9
        },
        "messages": [{"role": "user", "content": [{"text": prompt_string}]}]
    })
    
    start_time = time.perf_counter()
    response = client.invoke_model(
        modelId="amazon.nova-micro-v1:0", 
        contentType="application/json", 
        accept="application/json", 
        body=body
    )
    duration_ms = (time.perf_counter() - start_time) * 1000
    
    response_body = json.loads(response.get("body").read())

   # Slice into array element [0] to extract the text string from the Amazon Nova schema
    output_text = response_body["output"]["message"]["content"][0]["text"]
    output_tokens = response_body.get("usage", {}).get("outputTokens", 1)
    
    tokens_per_second = output_tokens / (duration_ms / 1000) if duration_ms > 0 else 0.0
    return output_text, round(duration_ms, 2), round(tokens_per_second, 2)

def run_cloud_rag_inference_suite():
    print("============================================================= ")
    print(" LAUNCHING PRODUCTION END-TO-END CLOUD RAG INFERENCE EVALUATOR")
    print("============================================================= \n")
    
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    conn = psycopg2.connect(DB_CONN_STR)
    cur = conn.cursor()
    
    rag_benchmark_records = []
    
    for idx, question in enumerate(BENCHMARK_QUERIES):
        target_table = f"aws_sec_table_{idx + 1}"
        print(f"Processing RAG Query #{idx+1:02d} targeting [{target_table}]...")
        
        # Step 1: Compute cloud embedding coordinates space
        query_vector = get_titan_embedding(bedrock_client, question)
        
        # Step 2: Database Retrieval via HNSW Graph Index (O(log N))
        start_db = time.perf_counter()
        cur.execute(
            f"SELECT text_content FROM {target_table} ORDER BY embedding <=> %s::vector LIMIT 1;", 
            (query_vector,)
        )
        context_block = cur.fetchone()
        db_latency_ms = (time.perf_counter() - start_db) * 1000
        
        # Format strict context constraint prompt window
        injected_context = context_block[0] if context_block else "No context found."
        rag_prompt = (
            f"Context: {injected_context}\n"
            f"Question: {question}\n"
            f"Answer the question factually using only the context provided above. If not present, say you don't know."
        )
        
        # Step 3: Cloud LLM Inference Generation Pass
        response_text, generation_latency_ms, tokens_per_sec = query_nova_micro(bedrock_client, rag_prompt)
        
        total_loop_latency_ms = db_latency_ms + generation_latency_ms
        
        print(f" Retrieval: {db_latency_ms:5.1f}ms | Generation: {generation_latency_ms:6.1f}ms | Total: {total_loop_latency_ms:6.1f}ms | Throughput: {tokens_per_sec:4.1f} t/s")
        
        rag_benchmark_records.append({
            "query_id": idx + 1,
            "user_query": question,
            "db_retrieval_latency_ms": round(db_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "total_rag_latency_ms": round(total_loop_latency_ms, 2),
            "tokens_per_second": tokens_per_sec,
            "model_response": response_text[:100] + "..."
        })
        
    # Write empirical performance logs directly to a CSV file
    with open(CSV_OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "user_query", "db_retrieval_latency_ms", "generation_latency_ms", "total_rag_latency_ms", "tokens_per_second", "model_response"])
        writer.writeheader()
        writer.writerows(rag_benchmark_records)
        
    cur.close()
    conn.close()
    print(f"\n Verification Complete! Live cloud RAG inference logs compiled inside '{CSV_OUTPUT_FILE}'.")

if __name__ == "__main__":
    run_cloud_rag_inference_suite()
