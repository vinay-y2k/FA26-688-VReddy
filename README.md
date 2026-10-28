# Hybrid Edge-Cloud RAG Semantic Router & Vector Database Performance Harness

An enterprise-grade, privacy-compliant, multi-environment Retrieval-Augmented Generation (RAG) platform and benchmarking framework. This repository isolates database retrieval complexities from underlying hardware architectures across a decentralized edge matrix and an auto-scaling public cloud tier.
To bypass shared memory buffer cache pollution anomalies that mask true algorithmic complexity curves, the system streams unstructured data records from the Hugging Face PleIAs/SEC corpus and splits them cleanly across 30 distinct, cache-isolated tables containing 10,000 rows each (300,000 total vectors).

## Architecture Overview

The system features an intelligent Semantic Intent Router middleware sitting at the application boundary. It computes real-time L₂-normalized vector dot products against frozen anchors to dynamically fork incoming queries, routing them down three distinct operational pathways before hitting downstream Large Language Models (LLMs):

<img width="902" height="328" alt="image" src="https://github.com/user-attachments/assets/3e556cd2-6e56-4885-b1ab-4479d8ff567e" />

• Path 1 (SEC Financial RAG): Triggers full vector database Approximate Nearest Neighbor (ANN) search loops using native pgvector operators (<=>) over the warm tables, forwarding context blocks to local or cloud LLMs.  
• Path 2 (NL-DBA Metadata Bypass): Intercepts administrative and database metric queries, executing a Zero-LLM-Cost Bypass that sweeps raw catalog statistics straight from disk . This provides a 1,000x latency reduction (~10ms vs ~15s).  
• Path 3 (Fallback General Chat): Catches ambiguous queries or generic prompts via a strict Relative Confidence Delta Gate (< 0.05) to safely prevent semantic misfires.

## Repository Structure & Core Manifest

<img width="985" height="768" alt="image" src="https://github.com/user-attachments/assets/304fba3b-8046-4171-8f34-f78b005a5cd4" />

## Environmental Specifications

## Decentralized Edge Frontier

• Hardware Accelerator: Intel Core Ultra 7 155H with integrated Intel Arc GPU (Vulkan Compute Framework active).  
• Vector Database: Local Docker container running PostgreSQL 16 + pgvector.  
• Embeddings Space: nomic-embed-text (768 Dimensions, un-normalized natively).  
• Local LLM Engine: Ollama managed 4-bit quantized Qwen-2.5-3B-Instruct locked in VRAM registers.

## Production Cloud Frontier

• Managed Vector Store: Amazon Aurora Serverless v2 cluster (Engine Version 16.3, constrained strictly to a 0.5 to 2.0 ACU scaling envelope).  
• Cloud Embeddings API: Amazon Titan Text Embeddings v2 via Amazon Bedrock (Configured to a native maximum 1024-dimensional boundary, L₂-normalized).  
• Cloud LLM Inference: Amazon Nova Micro via Amazon Bedrock via serverless API channels ($0.035 per 1M input tokens configuration).

## Deployment Operations & Execution Sequence

## Phase 1: Local Environment

1. Setup and run Postgres DB inside a Docker VM: powershell  
   docker run --name local-vector-db -e POSTGRES_DB=aifootprint -e POSTGRES_USER=db_admin_root -e POSTGRES_PASSWORD="xxxxx" -p 5432:5432 -d pgvector/pgvector:pg16
2. Install Ollama software from https://ollama.com/download and install below AI models  
   ollama pull nomic-embed-text  
   ollama pull qwen2.5:3b  
   Overwrite default Ollama parallel execution parameters and boot your local server instance safely: powershell  
   $env:OLLAMA_NUM_GPU="99;$env:OLLAMA_NUM_PARALLEL="999";$env:OLLAMA_MAX_LOADED_MODELS="2";$env:OLLAMA_FLASH_ATTENTION="1"
   ;$env:OLLAMA_VULKAN="1";$env:GGML_VK_VISIBLE_DEVICES="0"  
   ollama serve
3. Ingest Data - Stream dataset from hugging face, data chunking, generate vector embeddings, create DB table, populate it,create HNSW
   index: powershell  
   Populate .env file with Variables(DB_NAME,DB_USERNAME,DB_PASSWORD)
   python src/local/ingestion_pipeline_final.py
4. Evaluate local DB performance and plot captured data: powershell  
   python src/local/benchmarking_runner_final.py
   python src/local/vector_search_benchmarks.py
5. Generate local RAG Inference and plot captured data: powershell  
   python src/local/run_inference_benchmarks.py
   python src/local/plot_inference_timing.py

## Phase 2: Secure Public Cloud Provisioning

1. Inject your master database credentials token directly into local memory :powershell  
   cd terraform  
   $env:TF_VAR_db_password="YourCompliantSecureSecretPassword2026!"
2. Initialize and deploy your managed serverless cloud matrix via terraform :powershell  
   terraform init  
   terraform apply -auto-approve
3. Ingest Data - Stream dataset from hugging face, data chunking, generate vector embeddings, create DB table, populate it, create HNSW
   index: AWS Cloud shell to avoid network delays  
   Copy the output cluster database string endpoint printed to your console, paste it into the DB_HOST variable string inside cloud_bulk_ingest.py script.  
   Open AWS Cloud shell and upload cloud_bulk_ingest.py  
   export TF_VAR_db_password="YourCompliantSecureSecretPassword2026!"  
   python cloud_bulk_ingest.py

## Phase 3: Capturing Telemetry Curves & Fiscal Tear-Down

1. Run DB benchmarking queries: powershell  
   python src/cloud/cloud_benchmarker.py
2. Run RAG inference: powershell  
   python src/cloud/cloud_rag_evaluator.py
3. Generate graphs :powershell  
   python src/cloud/plot_global_comparison_final.py  
   python src/cloud/plot_cloud_rag.py
4. Execute automated teardown routine:powershell  
   cd src/terraform/  
   terraform destroy -auto-approve

## Empirical Metrics & Systems Insights

## Vector Search Complexity Acceleration Profiles

The system documents the exact relative speedup multiplier of fine-tuning an HNSW index graph (M=16, ef_construction=64) over brute-force unindexed sequential scans O(N) --> O(log N):  
• Local Sandbox (768-Dim): Brute Force: 285.31 ms | HNSW Index: 69.7 ms → 4.1x Latency Compression.  
• AWS Production (1024-Dim): Brute Force: 4,573.64 ms | HNSW Index: 108.85 ms → 42.0x Performance Acceleration.

**Insight**: Unindexed sequential vector operations inside distributed cloud architectures trigger high page pre-fetching penalties over virtualized storage arrays. Fine-tuning the HNSW proximity graph completely neutralizes hardware latency inflation, locking search lookups to a tight 108.85 ms turnaround window even when vector geometric dimensions double.

## Multi-Path Routing & Auto-Scaling Telemetry

• Path 2 NL-DBA Bypass Efficiency: Intercepting infrastructure lookups at the application boundary resolves metadata queries in ~10.4 ms to 32.6 ms with exactly 0.0 tokens/second of LLM overhead.  
• Aurora Cold-Start Signature: End-to-end RAG metrics capture a clear serverless auto-scaling cold-start curve on initial queries (spiking to 8,295 ms at 0.5 ACU baseline parameters) before stabilizing smoothly at ~5271 ms text generation latency once the cluster reaches its warm 2.0 ACU compute ceiling.
