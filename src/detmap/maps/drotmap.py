lic_ = """
   Copyright 2026 Richard Tjörnhammar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

"""
Ultra-High Dim Space-Filling Ensemble Map
Improved version

Improvements:
- random rotation ensemble
- Morton (Z-order) indexing (faster than Hilbert)
- convolution smoothing (GPU optimized)
- float32 compute
- better JAX fusion
"""

import os
os.environ["JAX_ENABLE_X64"] = "False"

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import time


# -------------------------------------------------------
# RANDOMIZED SVD
# -------------------------------------------------------

def randomized_svd_chunked(X, rank, oversample=8, n_iter=2, key=jax.random.PRNGKey(0), chunk_size=10000):
    N, D = X.shape
    l = rank + oversample

    Omega = jax.random.normal(key, (D, l), dtype=X.dtype)

    def matmul_chunked(A, B):
        out = []
        for i in range(0, A.shape[0], chunk_size):
            out.append(A[i:i+chunk_size] @ B)
        return jnp.vstack(out)

    Y = matmul_chunked(X, Omega)

    for _ in range(n_iter):
        Y = matmul_chunked(X, matmul_chunked(X.T, Y))

    Q, _ = jnp.linalg.qr(Y)

    B = Q.T @ X
    U_hat, S, Vt = jnp.linalg.svd(B, full_matrices=False)

    U = Q @ U_hat

    return U[:, :rank], S[:rank], Vt[:rank, :]


def svd_reduce(X, rank, key):
    U, S, V = randomized_svd_chunked(X, rank, key=key)
    return X @ V.T


def randomized_pca_jax(X, n_components=3, n_oversamples=10, n_iter=2,
                       random_key=jax.random.PRNGKey(42)):
    """
    Correct randomized PCA using randomized SVD.

    Args:
        X: array of shape (n_samples, n_features)
        n_components: number of components to keep
        n_oversamples: oversampling parameter (Halko et al. recommend 10)
        n_iter: number of power iterations (1-2 usually sufficient)
        random_key: JAX random key

    Returns:
        X_pca: projected data (n_samples, n_components)
        components: principal components (n_components, n_features)
        explained_variance: variance explained
    """
    # Center the data
    X_centered = X - jnp.mean(X, axis=0)

    n_samples, n_features = X_centered.shape
    l = n_components + n_oversamples

    # Generate random matrix
    key, subkey = jax.random.split(random_key)
    Omega = jax.random.normal(subkey, (n_features, l))

    # Stage A: Find approximate range
    Y = X_centered @ Omega

    # Power iteration (improves accuracy)
    for _ in range(n_iter):
        Y = X_centered @ (X_centered.T @ Y)

    # Orthonormalize
    Q, _ = jnp.linalg.qr(Y)

    # Stage B: Compute SVD on smaller matrix
    B = Q.T @ X_centered
    U_tilde, S, Vt = jnp.linalg.svd(B, full_matrices=False)

    # Recover left singular vectors of X
    U = Q @ U_tilde

    # Project data
    X_pca = X_centered @ Vt.T[:, :n_components]

    # Explained variance
    explained_variance = (S[:n_components]**2) / (n_samples - 1)

    return X_pca, Vt[:n_components], explained_variance

# Benchmark to verify correctness
def verify_pca_correctness():
    """Compare randomized PCA with full SVD PCA."""
    from sklearn.datasets import make_blobs
    from sklearn.decomposition import PCA
    import numpy as np

    # Generate test data
    X, _ = make_blobs(n_samples=500, n_features=20, centers=3, random_state=42)
    X_jax = jnp.array(X)

    # Scikit-learn PCA (ground truth)
    pca_sk = PCA(n_components=3)
    X_pca_sk = pca_sk.fit_transform(X)

    # Our randomized PCA
    X_pca_jax, components_jax, ev_jax = randomized_pca_jax(X_jax, n_components=3)

    # Compare projections (they should be similar up to sign)
    correlation = np.corrcoef(X_pca_sk[:, 0], np.array(X_pca_jax[:, 0]))[0, 1]
    print(f"Correlation with sklearn PCA (first component): {correlation:.6f}")
    print(f"Should be very close to 1.0 (or -1.0 if sign flipped)")

    return abs(correlation) > 0.99


# -------------------------------------------------------
# RANDOM ROTATION (improves locality)
# -------------------------------------------------------

def random_rotation(key, X):
    D = X.shape[1]

    M = jax.random.normal(key, (D, D), dtype=X.dtype)
    Q, _ = jnp.linalg.qr(M)

    return X @ Q


# -------------------------------------------------------
# MORTON (Z-ORDER) INDEX
# -------------------------------------------------------

@partial(jax.jit, static_argnames=("bits",))
def morton_index_nd(coords, bits):

    coords = coords.astype(jnp.uint32)
    dims = coords.shape[0]

    h = jnp.uint64(0)

    for b in range(bits):
        for d in range(dims):
            bit = (coords[d] >> b) & 1
            h |= bit << (b * dims + d)

    return h


@partial(jax.jit, static_argnames=("bits",))
def morton_index_batch(points, bits):
    return jax.vmap(lambda p: morton_index_nd(p, bits))(points)


# -------------------------------------------------------
# BUILD GRID + ORDER
# -------------------------------------------------------

@partial(jax.jit, static_argnames=("bits",))
def build_morton_map(X, mins, maxs, bits):

    grid = jnp.floor(
        (X - mins) / (maxs - mins + 1e-9) * (2 ** bits - 1)
    ).astype(jnp.uint32)

    morton = morton_index_batch(grid, bits)

    order = jnp.argsort(morton)

    return X[order], order


# -------------------------------------------------------
# FAST SMOOTHING (CONVOLUTION)
# -------------------------------------------------------

def hilbert_smooth_fast(X_sorted, window):

    kernel = jnp.ones((window,), dtype=X_sorted.dtype) / window

    def smooth_dim(x):
        return jsp.signal.convolve(x, kernel, mode="same")

    return jax.vmap(smooth_dim, in_axes=1, out_axes=1)(X_sorted)


# -------------------------------------------------------
# ENSEMBLE MAP
# -------------------------------------------------------

def hilbert_ensemble_map(
    X,
    reduced_dims=64,
    bits=6,
    window=32,
    ensemble_size=4,
    key=jax.random.PRNGKey(0),
):

    N, D = X.shape

    # dimensionality reduction
    X_low = svd_reduce(X, reduced_dims, key)

    # normalization once
    mins = jnp.min(X_low, axis=0)
    maxs = jnp.max(X_low, axis=0)

    embeddings = []

    subkey = key

    for i in range(ensemble_size):

        subkey, rot_key, perm_key = jax.random.split(subkey, 3)

        # random rotation
        X_rot = random_rotation(rot_key, X_low)

        # random permutation
        perm = jax.random.permutation(perm_key, X_rot.shape[1])

        X_perm = X_rot[:, perm]

        # space filling curve ordering
        X_sorted, order = build_morton_map(X_perm, mins, maxs, bits)

        # smoothing
        X_smooth = hilbert_smooth_fast(X_sorted, window)

        inverse_order = jnp.zeros_like(order)
        inverse_order = inverse_order.at[order].set(jnp.arange(N))

        embeddings.append(X_smooth[inverse_order])

    return jnp.mean(jnp.stack(embeddings), axis=0)


# -------------------------------------------------------
# SYNTHETIC TEST DATA
# -------------------------------------------------------

def generate_blobs(key, n_points=100000, dims=1000, n_clusters=5, spread=0.5):

    subkey = key

    points_per_cluster = n_points // n_clusters

    clusters = []

    for i in range(n_clusters):

        subkey, k1, k2 = jax.random.split(subkey, 3)

        center = jax.random.normal(k1, (dims,)) * 5.0

        cluster = center + jax.random.normal(k2, (points_per_cluster, dims)) * spread

        clusters.append(cluster)

    return jnp.vstack(clusters).astype(jnp.float32)


# -------------------------------------------------------
# TEST
# -------------------------------------------------------

def run_test():

    key = jax.random.PRNGKey(0)

    N = 100000
    D = 1000
    clusters = 5

    print(f"Generating {clusters} clusters with {N} points in {D} dimensions")

    X = generate_blobs(key, N, D, clusters)

    print("Input shape:", X.shape)

    t0 = time.time()

    emb = hilbert_ensemble_map(
        X,
        reduced_dims=64,
        bits=6,
        window=32,
        ensemble_size=6,
        key=key
    )

    jax.block_until_ready(emb)

    t1 = time.time()

    print("Embedding shape:", emb.shape)
    print("Time:", t1 - t0, "seconds")

    try:

        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,6))

        subsample = emb[:5000]

        plt.scatter(subsample[:,0], subsample[:,1], s=2)

        plt.title("Space Filling Ensemble Map")

        plt.show()

    except Exception:

        print("Matplotlib not available.")


# -------------------------------------------------------
# CSV TEST
# -------------------------------------------------------

def run_data_test(filename=None,df=None):

    if filename is not None:
        import pandas as pd
        df = pd.read_csv(filename, sep="\t")
        df = df.select_dtypes(include="number")

    X = jnp.array(df.values, dtype=jnp.float32)

    print("Loaded data shape:", X.shape)

    key = jax.random.PRNGKey(123)

    t0 = time.time()

    embedding = hilbert_ensemble_map(
        X,
        reduced_dims=min(64, X.shape[1]),
        bits=6,
        window=32,
        ensemble_size=4,
        key=key
    )

    jax.block_until_ready(embedding)

    t1 = time.time()

    print("Embedding shape:", embedding.shape)
    emb=embedding
    print("Time:", t1 - t0)
    #emb,_,_ = randomized_pca_jax(embedding)

    # Optional: plot first 2 dims
    try:

        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(emb[:,0], emb[:,1], s=3, alpha=0.7)
        plt.title("Hilbert Ensemble Map (CSV data)")
        plt.show()
    except Exception:
        print("Matplotlib not available, skipping plot.")


# -------------------------------------------------------

"""DROT Map implementation."""

from ..base import BaseMap
import numpy as np

class DetMap(BaseMap):
    """
    Deterministic Rotation Map.
    
    This is the main embedding algorithm for detmap.
    """
    
    def __init__(self, n_components=2, ensemble_size=10, random_state=None):
        super().__init__(n_components, random_state)
        self.ensemble_size = ensemble_size
        
    def fit(self, X, y=None):
        X = self._check_array(X)
        # Your DROT map implementation here
        # Can use from ..quantification import multivariate_aligned_pca
        return self
    
    def transform(self, X):
        X = self._check_array(X)
        # Transform implementation
        pass


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("analytes.csv",sep='\t').iloc[:,1:]
    run_data_test(df=df)
