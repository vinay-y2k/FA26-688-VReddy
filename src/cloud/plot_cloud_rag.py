# pip install pandas matplotlib
import pandas as pd
import matplotlib.pyplot as plt

def generate_stacked_cloud_rag_chart():
    csv_file = "cloud_rag_inference_benchmarks.csv"
    output_image = "cloud_end_to_end_rag_breakdown.png"
    
    try:
        # Load empirical cloud performance logs
        df = pd.read_csv(csv_file)
        
        plt.figure(figsize=(14, 7))
        
        # RENDER STACKED BARS
        # Base Layer: Database Retrieval Latency via HNSW Index
        plt.bar(df["query_id"], df["db_retrieval_latency_ms"], 
                color="#2ca02c", alpha=0.9, label="Database Retrieval (Aurora HNSW Index)")
        
        # Stacked Layer: Text Generation Latency via Amazon Nova Micro
        plt.bar(df["query_id"], df["generation_latency_ms"], 
                bottom=df["db_retrieval_latency_ms"],
                color="#1f77b4", alpha=0.9, label="Text Generation (Amazon Nova Micro)")
        
        # ADD MEAN TREND LINE FOR TOTAL E2E LATENCY
        plt.plot(df["query_id"], df["total_rag_latency_ms"], 
                 color="#d62728", linestyle="--", linewidth=1.5, marker="o", markersize=4, label="Total End-to-End RAG Latency")
        
        # Typography and Layout Configuration
        plt.title("Production Scale Cloud RAG Profile: Stacked Latency Components and System Turnaround Bounds", fontsize=12, fontweight="bold", pad=20)
        plt.xlabel("Standardized Evaluation Query Identifier ID (1 to 30)", fontsize=10, labelpad=12)
        plt.ylabel("Inference Execution Footprint (Milliseconds)", fontsize=10, labelpad=12)
        
        # Formatting Grid Lines
        plt.grid(True, axis='y', linestyle=":", alpha=0.5)
        plt.xticks(df["query_id"])
        
        # Include data label values on top of select bars to showcase the metric distribution
        mean_total = df["total_rag_latency_ms"].mean()
        plt.axhline(y=mean_total, color="#d62728", linestyle=":", alpha=0.7, 
                    label=f"Mean E2E System Latency ({mean_total:.1f} ms)")
        
        plt.legend(loc="upper right", fontsize=10)
        
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"🎉 Success! Publication-grade stacked RAG chart exported to '{output_image}'.")
        
    except FileNotFoundError:
        print(f"❌ Error: '{csv_file}' not found. Ensure your cloud_rag_evaluator.py script has completed successfully.")

if __name__ == "__main__":
    generate_stacked_cloud_rag_chart()
