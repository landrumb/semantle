"""Construct an fbin for gensim embeddings, along with queries, vocab, and ground truth."""
import gensim.downloader as api
import numpy as np
import sys
from pathlib import Path
import subprocess
import re

from utils import numpy_to_fbin

def words_to_file(words, file_path):
    with open(file_path, "w") as file:
        for word in words:
            file.write(word + "\n")
            
def is_lowercase_word(word: str) -> bool:
    """returns true if a word consists only of lowercase letters and underscores"""
    return re.match("^[a-z_]+$", word) is not None
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python embeddings_to_fbin.py <gensim embedding name> [crop length] [--lowercase_only]")
        sys.exit(1)
        
    embeddings = sys.argv[1]
    
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        crop_length = int(sys.argv[2])
    else:
        crop_length = None
        
    if "--lowercase_only" in sys.argv:
        lowercase = True
    else:
        lowercase = False
    
    download_dir = Path(f"data/{embeddings}")
    if crop_length is not None:
        download_dir = download_dir.with_name(f"{download_dir.name}_{crop_length}")
    if lowercase:
        download_dir = download_dir.with_name(f"{download_dir.name}_lowercase")
    
    
    if download_dir.exists():
        print("Embeddings already downloaded, but running again.")
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {embeddings}...")
    model = api.load(embeddings)
    vectors = model.vectors
    vocab = model.index_to_key
    
    # crop then lowercase; not necessarily the best order
    if crop_length is not None:
        print(f"Cropping to {crop_length} words...")
        vectors = vectors[:crop_length]
        vocab = vocab[:crop_length]
    
    if lowercase:
        print("Filtering to lowercase words...")
        lowercase_indices = [i for i, word in enumerate(vocab) if is_lowercase_word(word)]
        vectors = vectors[lowercase_indices]
        vocab = [vocab[i] for i in lowercase_indices]
        
    # normalize vectors
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    
    print("Saving embeddings...")
    numpy_to_fbin(vectors, download_dir / "base.fbin")
    words_to_file(vocab, download_dir / "vocab.txt")
    
    nq = 5000
    print(f"Saving {nq} queries...")
    query_indices = np.random.choice(len(vocab), nq, replace=False)
    query_indices.sort()
    
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
                    "-k", "100",
                    "-dist_func", "mips",
                    "-data_type", "float"])
    