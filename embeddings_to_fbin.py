"""Construct an fbin for gensim embeddings"""
import gensim.downloader as api
import numpy as np

def embeddings_to_fbin(fbin_path=None,
                       embeddings="glove-wiki-gigaword-50",
                       normalize=True):
    if fbin_path is None:
        fbin_path = f"{embeddings}.fbin"
    
    model = api.load(embeddings)
    n, d = model.vectors.shape
    
    
    if normalize:
        print("Normalizing vectors...")
        vectors = model.vectors / np.linalg.norm(model.vectors, axis=1)[:, np.newaxis]
    else:
        vectors = model.vectors
    
    print(f"Writing {n} vectors of dimension {d} to {fbin_path}")
    with open(fbin_path, "wb") as fbin_file:
        np.array([n, d], dtype=np.int32).tofile(fbin_file)
        vectors.astype(np.float32).tofile(fbin_file)
    print("Done.")
    
    
if __name__ == "__main__":
    embeddings_to_fbin(normalize=False)
    