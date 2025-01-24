"""Construct an fbin for gensim embeddings, along with queries, vocab, and ground truth."""
import gensim.downloader as api
import numpy as np
import sys
from pathlib import Path
import subprocess

from utils import numpy_to_fbin

def words_to_file(words, file_path):
    with open(file_path, "w") as file:
        for word in words:
            file.write(word + "\n")
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embeddings_to_fbin.py <gensim embedding name>")
        sys.exit(1)
        
    embeddings = sys.argv[1]
    
    if Path(f"data/{embeddings}").exists():
        print("Embeddings already downloaded, but running again.")
    
    download_dir = Path(f"data/{embeddings}")
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {embeddings}...")
    model = api.load(embeddings)
    vectors = model.vectors
    vocab = model.index_to_key
    
    print("Saving embeddings...")
    numpy_to_fbin(vectors, download_dir / "base.fbin")
    words_to_file(vocab, download_dir / "vocab.txt")
    
    nq = 10000
    print(f"Saving {nq} queries...")
    query_indices = np.random.choice(min(len(vocab) // 50, 30000), nq, replace=False)
    query_words = [vocab[i] for i in query_indices]
    query_vectors = vectors[query_indices]
    
    numpy_to_fbin(query_vectors, download_dir / "query.fbin")
    words_to_file(query_words, download_dir / "query.txt")
    
    print("Computing (and saving) ground truth...")
    
    if not Path("ParlayANN/data_tools/compute_groundtruth").exists():
        print("Please compile ParlayANN/data_tools/compute_groundtruth (run make in ParlayANN/data_tools/)")
        sys.exit()
    
    subprocess.run(["ParlayANN/data_tools/compute_groundtruth",
                    "-base_path", download_dir / "base.fbin",
                    "-query_path", download_dir / "query.fbin",
                    "-gt_path", download_dir / "GT",
                    "-k", "1000",
                    "-dist_func", "Euclidian",
                    "-data_type", "float"])
    