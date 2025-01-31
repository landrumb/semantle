from ParlayANN.python import wrapper as wp

import sys
from pathlib import Path
import numpy as np

from utils import graph_file_to_list_of_lists, read_vocab, fbin_to_numpy
from beam_search import eager_beam_search, beam_search
from tqdm import tqdm

GRAPH_TYPES = ["pynndescent", "vamana", "hcnng"]

VAMANA_PARAMS = {
    "R": 8,
    "L": 75,
    "alpha": .99,
    "two_pass": True
}

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
        wp.build_vamana_index("mips", "float", data_dir / "base.fbin", data_dir / "outputs" / "vamana", **VAMANA_PARAMS)
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
    
_, vocab = read_vocab(data_dir / "vocab.txt")
_, query_vocab = read_vocab(data_dir / "query.txt")

query_indices = [vocab.index(word) for word in query_vocab]

vectors = fbin_to_numpy(data_dir / "base.fbin")

visited_counts = []
compared_counts = []

# print("eager beam search:")
# for idx in tqdm(query_indices):
#     visited, compared = eager_beam_search(graph, vectors, 0, idx)
#     visited_counts.append(len(visited))
#     compared_counts.append(len(compared))

# print(f"average visited: {np.mean(visited_counts)}")
# print(f"average compared: {np.mean(compared_counts)}")

# visited_counts = []
# compared_counts = []

print("beam search:")
for idx in tqdm(query_indices[:1000]):
    visited, compared = eager_beam_search(graph, vectors, 0, idx)
    visited_counts.append(len(visited))
    compared_counts.append(len(compared))

print(f"average visited: {np.mean(visited_counts)}")
print(f"average compared: {np.mean(compared_counts)}")

converged = np.array(visited_counts) < 1000
print(f"recall: {np.mean(converged)}")

log_file = data_dir / "outputs" / f"{graph_type}_log.csv"

if not log_file.exists():
    with open(log_file, "w") as f:
        f.write(f"visited,compared,{",".join(VAMANA_PARAMS.keys())}\n")
        
with open(log_file, "a") as f:
    f.write(f"{np.mean(visited_counts)},{np.mean(compared_counts)},{','.join(str(value) for value in VAMANA_PARAMS.values())}\n")