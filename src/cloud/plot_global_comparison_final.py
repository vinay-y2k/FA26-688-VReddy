import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_global_comparison_chart():
    local_csv = "vector_search_benchmarks.csv"
    cloud_csv = "cloud_search_benchmarks.csv"
    output_image = "global_latency_frontier_analysis_final.png"
    
    try:
        # Load empirical log datasets
        df_local = pd.read_csv(local_csv)
        df_cloud = pd.read_csv(cloud_csv)
        
        # 1. Initialize the figure and axis with clear dimensional footprints
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 2. Configure grouped bar layout parameters
        x = np.arange(len(df_local))  # Positional index base for the 30 queries
        width = 0.20                  # Individual width allocated to each of the 4 metrics
        
        # Construct clean label identifiers for the x-axis: "Q1", "Q2", etc.
        queries = [f"Q{qid}" for qid in df_local["query_id"]]
        
        # 3. Plot each execution footprint with precise geometric offsets
        rects1 = ax.bar(x - (1.5 * width), df_local["latency_unindexed_ms"], width, 
                        label="Local: Sequential Scan (768-Dim)", color="#d62728", alpha=0.7)
        
        rects2 = ax.bar(x - (0.5 * width), df_local["latency_indexed_ms"], width, 
                        label="Local: HNSW Index Scan (768-Dim)", color="#1f77b4")
        
        rects3 = ax.bar(x + (0.5 * width), df_cloud["latency_unindexed_ms"], width, 
                        label="Cloud: Sequential Scan (1024-Dim)", color="#ff7f0e", alpha=0.7)
        
        rects4 = ax.bar(x + (1.5 * width), df_cloud["latency_indexed_ms"], width, 
                        label="Cloud: HNSW Index Scan (1024-Dim)", color="#2ca02c")
        
        # 4. Apply strict academic document formatting properties
        ax.set_yscale('log') # Maintain logarithmic scaling to handle cross-hardware range gaps
        
        ax.set_xticks(x)
        # Rotate 45 degrees and align right to ensure 30 long strings stay clearly visible
        ax.set_xticklabels(queries, rotation=45, ha="right", fontsize=9)
        
        plt.title("Boundary Frontier Analysis: Latency Scalability of Edge (768-Dim) vs. Cloud (1024-Dim) RAG Networks", 
                  fontsize=12, fontweight="bold", pad=20)
        plt.xlabel("Standardized Evaluation Query Identifier ID (1 to 30)", fontsize=10, labelpad=12)
        plt.ylabel("Core Search Execution Footprint (Milliseconds)", fontsize=10, labelpad=12)
        
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)
        
        # Clean structural padding adjustments
        plt.tight_layout()
        
        # Save as a high-density, sharp image file for your Word document appendix
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"Success! Comparative thesis chart exported directly to '{output_image}'.")
        
    except FileNotFoundError as e:
        print(f"Error: Data logs missing. Verify both CSV files are present. details: {e}")

if __name__ == "__main__":
    generate_global_comparison_chart()
