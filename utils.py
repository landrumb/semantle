"""misc utility functions"""

import numpy as np

def numpy_to_fbin(vectors, fbin_path):
    n, d = vectors.shape
    with open(fbin_path, "wb") as fbin_file:
        np.array([n, d], dtype=np.int32).tofile(fbin_file)
        vectors.astype(np.float32).tofile(fbin_file)
