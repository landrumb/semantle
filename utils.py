"""misc utility functions"""

import numpy as np
from pathlib import Path

def numpy_to_fbin(vectors, fbin_path):
    """writes a 2d numpy array to a .fbin file"""
    n, d = vectors.shape
    with open(fbin_path, "wb") as fbin_file:
        np.array([n, d], dtype=np.int32).tofile(fbin_file)
        vectors.astype(np.float32).tofile(fbin_file)

def fbin_to_numpy(fbin_path):
    """reads a 2d numpy array from a .fbin file"""
    with open(fbin_path, "rb") as fbin_file:
        n, d = np.fromfile(fbin_file, dtype=np.int32, count=2)
        return np.fromfile(fbin_file, dtype=np.float32).reshape(n, d)
    
def graph_file_to_list_of_lists(graph_file):
    """reads a parlay graph file and returns a list of lists representing out neighborhoods"""
    with open(graph_file, "rb") as f:
        num_points, max_degree = np.fromfile(f, dtype=np.int32, count=2)
        print(f"reading {num_points} points with max degree {max_degree}")
        
        degrees = np.fromfile(f, dtype=np.int32, count=num_points)
        out_neighborhoods = []
        for degree in degrees:
            out_neighborhoods.append(np.fromfile(f, dtype=np.int32, count=degree))
        
        remaining = np.fromfile(f, dtype=np.int32)
        assert len(remaining) == 0, f"file has {len(remaining)} remaining values after reading, expected 0"
        
        return out_neighborhoods
    
def read_vocab(vocab_dir):
    """returns a word_to_idx dict and idx_to_word list from a vocab directory"""
    if type(vocab_dir) == str:
        vocab_dir = Path(vocab_dir)
    
    word_to_idx = {}
    idx_to_word = []
    
    with open(vocab_dir) as file:
        for idx, line in enumerate(file):
            word = line.strip()
            word_to_idx[word] = idx
            idx_to_word.append(word)
            
    return word_to_idx, idx_to_word