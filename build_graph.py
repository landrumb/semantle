from ParlayANN.python import wrapper as wp

import sys
import os
from pathlib import Path
import numpy as np

from utils import graph_file_to_list_of_lists

GRAPH_TYPES = ["pynndescent", "vamana", "hcnng"]

if len(sys.argv) < 2:
    print("Usage: python build_graph.py <embedding name> <graph type>")
    sys.exit(1)
    
embeddings = sys.argv[1]
graph_type = sys.argv[2]

data_dir = Path(f"data/{embeddings}")

if not Path(f"data/{embeddings}").exists():
    print("Embeddings not found. Run embeddings_to_fbin.py first.")
    sys.exit(1)
    
if graph_type not in GRAPH_TYPES:
    print(f"Graph type must be one of {GRAPH_TYPES}")
    sys.exit(1)
    
if not (data_dir / "outputs").exists():
    (data_dir / "outputs").mkdir()
    

if '--dont-build' not in sys.argv:
    if graph_type == "pynndescent":
        wp.build_pynndescent_index("Euclidian", "float", data_dir / "base.fbin", data_dir / "outputs" / "pynndescent", 40, 10, 100, 1.2, .05)
    elif graph_type == "vamana":
        wp.build_vamana_index("mips", "float", data_dir / "base.fbin", data_dir / "outputs" / "vamana", 8, 100, 0.99, False)
    elif graph_type == "hcnng":
        wp.build_hcnng_index("Euclidian", "float", data_dir / "base.fbin", data_dir / "outputs" / "hcnng", 40, 20, 1000)
    
print("Graph built. Testing recall...")

Index = wp.load_index("mips", "float", data_dir / "base.fbin", data_dir / "outputs" / graph_type)

neighbors, distances = Index.batch_search_from_string(str(data_dir / "query.fbin"), 10, 50, True, 10000)

Index.check_recall(str(data_dir / "query.fbin"), str(data_dir / "GT"), neighbors, 10)

graph = graph_file_to_list_of_lists(data_dir / "outputs" / graph_type)


print(f"{len(graph)} points")
print(f"{sum(len(neighbors) for neighbors in graph)} edges")
print(f"{sum(len(neighbors) for neighbors in graph) / len(graph)} average degree")

print("neighbor count distribution:")
percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    print(f"{p}%: {np.percentile([len(neighbors) for neighbors in graph], p)}")