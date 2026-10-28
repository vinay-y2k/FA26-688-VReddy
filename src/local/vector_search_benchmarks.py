import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get the path to the directory where this script lives
SCRIPT_DIR = Path(__file__).resolve().parent
# print(SCRIPT_DIR)


def generate_performance_curves():
    csv_file = SCRIPT_DIR /".."/"csv_image_files"/ "vector_search_benchmarks.csv"
    
    try:
        # Load empirical log matrices from your benchmarking run
        df = pd.read_csv(csv_file)

        # Create the figure and subplots
        fig, ax = plt.subplots(figsize=(10, 6))
        

        # Position the bars on the x-axis
        x = range(len(df))  # The label locations: [0, 1, 2]
        queries = [ f"Q{qid}" for qid in df["query_id"] ]
        width = 0.35                   # Width of each individual bar

       
        # 4. Plot each group, offsetting their x positions
        rects1 = ax.bar([pos - width/2 for pos in x] ,df["latency_unindexed_ms"] , width, label='Sequential Scan (O(N) Complexity)1', color='#d9534f')
        rects2 = ax.bar([pos + width/2 for pos in x],df["latency_indexed_ms"], width, label='HNSW Graph Index (O(log N) Complexity)', color='#5cb85c')


        ax.set_xticks(x)
        # Rotate labels by 45 degrees and align them to the right to prevent overlap
        ax.set_xticklabels(queries, rotation=45, ha="right", fontsize=9)
       
        plt.title("Computational Performance Scaling: Unindexed vs. Fine-Tuned Vector Spaces", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Standardized Query Identifier ID", fontsize=10)
        plt.ylabel("Execution Search Latency (Milliseconds)", fontsize=10)
        
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()
        
        
        # Save as a high-density, sharp image file for your Word document appendix
        output_image = SCRIPT_DIR / ".." / "csv_image_files" / "vector_search_performance_curves_a5.png"
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"Success! Publication-grade performance chart exported directly to '{output_image}'.")
        
    except FileNotFoundError:
        print(f"Error: '{csv_file}' not located. Please run 'benchmarking_runner.py' first.")

if __name__ == "__main__":
    generate_performance_curves()
