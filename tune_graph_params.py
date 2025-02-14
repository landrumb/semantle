from ParlayANN.python import wrapper as wp
from utils import graph_file_to_list_of_lists, read_vocab, fbin_to_numpy
from beam_search import eager_beam_search, beam_search

import optuna
import sys
from pathlib import Path
import numpy as np

embeddings = "word2vec-google-news-300_50000_lowercase"

data_dir = Path(f"data/{embeddings}")

_, vocab = read_vocab(data_dir / "vocab.txt")
_, query_vocab = read_vocab(data_dir / "query.txt")

query_indices = [vocab.index(word) for word in query_vocab]

vectors = fbin_to_numpy(data_dir / "base.fbin")


def objective(trial):
    VAMANA_PARAMS = {
        "R": trial.suggest_int("R", 4, 16),  # Graph degree
        "L": trial.suggest_int("L", 25, 200),  # Search width
        "alpha": trial.suggest_float("alpha", 0.75, 1.25, step=0.005),  # Expansion factor
        "two_pass": trial.suggest_categorical("two_pass", [True, False])  # Whether to use two-pass construction
    }

    wp.build_vamana_index("mips", "float", data_dir / "base.fbin", data_dir / "outputs" / "vamana", **VAMANA_PARAMS)
    
    graph = graph_file_to_list_of_lists(data_dir / "outputs" / "vamana")

    visited_counts = []
    compared_counts = []
    
    for idx in query_indices[:1000]:
        visited, compared = eager_beam_search(graph, vectors, 0, idx)
        visited_counts.append(len(visited))
        compared_counts.append(len(compared))

    avg_compared = np.mean(compared_counts)
    converged = np.array(visited_counts) < 1000
    
    recall = np.mean(converged)
    print(f"recall: {recall}, avg compared: {avg_compared}")

    return avg_compared  if recall > 0.995 else 1000

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=250)  # Adjust the number of trials as needed

print("Best hyperparameters:", study.best_params)