import pandas as pd
import matplotlib.pyplot as plt

def generate_inference_timing_chart():
    csv_file = "inference_timing_benchmarks.csv"
    output_image = "inference_latency_comparison_u1.png"
    
    try:
        # Load empirical log datasets
        df = pd.read_csv(csv_file)
        
        plt.figure(figsize=(12, 6))
        
        # Map out distinct academic color themes for each architectural path
        color_map = {
            "Path 1: Financial RAG": "#1f77b4",             # Crisp Blue
            "Path 2: NL-DBA Bypass": "#2ca02c",             # Fast Green
            "Path 3: Fallback": "#7f7f7f",                  # Muted Gray
            "Path 3: Fallback (Ambiguous Block)": "#ff7f0e" # Alert Orange
        }
        
        colors = [color_map.get(route, "#7f7f7f") for route in df["route_chosen"]]
        
        # Render horizontal bar layout for optimal scannability
        bars = plt.barh(df["prompt_id"], df["latency_ms"], color=colors, height=0.6, edgecolor="none")
        
        # Apply professional text and formatting configurations
        plt.title("Routing Architecture Performance Profiles: Inference Latency and Bypass Optimization Scaling", fontsize=12, fontweight="bold", pad=20)
        plt.xlabel("Total Response Latency Execution Footprint (Milliseconds)", fontsize=10, labelpad=12)
        plt.ylabel("Standardized Test Prompt Identifier ID", fontsize=10, labelpad=12)
        
        # Enforce logarithmic horizontal grid structures for wide scale differentials
        plt.xscale('log') 
        plt.grid(True, axis='x', linestyle=":", alpha=0.6, which="both")
        plt.yticks(df["prompt_id"])
        plt.gca().invert_yaxis()  # Keep Prompt 01 structured cleanly at the top line
        
        # Add a custom structural legend box
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', label='Path 1: Financial RAG (Heavy Compute)'),
            Patch(facecolor='#2ca02c', label='Path 2: Admin Bypass (Zero LLM Overhead)'),
            # Patch(facecolor='#ff7f0e', label='Path 3: Delta Gate Fallback (Safety Catcher)')
            Patch(facecolor='#ff7f0e', label='Path 3: Fallback (Ambiguous Block / Delta Gate)'),
            Patch(facecolor='#7f7f7f', label='Path 3: Fallback (Standard General Chat)')
        ]
        # plt.legend(handles=legend_elements, loc="lower right", fontsize=10)
        plt.legend(handles=legend_elements, loc="best", fontsize=10)
        # plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize = 10,ncols=2)

        # Annotate exact millisecond markers next to active data bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.15, bar.get_y() + bar.get_height()/2, f'{width:.1f} ms', 
                     va='center', ha='left', fontsize=8, color='#333333', fontweight="bold")

        # plt.tight_layout()    
        plt.savefig(output_image, dpi=300, bbox_inches="tight")
        print(f"Success! Publication-grade chart exported directly to '{output_image}'.")
        
    except FileNotFoundError:
        print(f" Error: '{csv_file}' data logs missing. Run your benchmark script first.")

if __name__ == "__main__":
    generate_inference_timing_chart()
