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
Ultra-High Dim Hilbert Ensemble Map (1M+ points, 10k+ dims)
Deterministic, GPU/TPU-compatible, cluster-preserving
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
from functools import partial
from jax import lax
import time

# -------------------------------
# CHUNKED RANDOMIZED SVD
# -------------------------------

def randomized_svd_chunked(X, rank, oversample=8, n_iter=2, key=jax.random.PRNGKey(0), chunk_size=10000):
    N, D = X.shape
    l = rank + oversample
    Omega = jax.random.normal(key, (D, l))
    
    def matmul_chunked(A, B):
        """Compute A @ B in row chunks"""
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

# -------------------------------
# HILBERT INDEX
# -------------------------------

@partial(jax.jit, static_argnames=("bits",))
def hilbert_index_nd(coords, bits):
    coords = coords.astype(jnp.uint64)
    n = coords.shape[0]
    x = coords
    def gray_step(i, x):
        bit = jnp.uint64(1) << (bits-1-i)
        def inner(j, x):
            cond = (x[j] & bit) != 0
            x = jnp.where(cond, x ^ (bit-1), x)
            t = (x[0] ^ x[j]) & (bit-1)
            x = x.at[0].set(x[0] ^ t)
            x = x.at[j].set(x[j] ^ t)
            return x
        return lax.fori_loop(1, n, inner, x)
    x = lax.fori_loop(0, bits, gray_step, x)
    def build_index(i, state):
        x, h = state
        b = bits-1-i
        digit = jnp.uint64(0)
        for d in range(n):
            digit |= ((x[d] >> b) & 1) << d
        h = (h << n) | digit
        return (x,h)
    _, h = lax.fori_loop(0, bits, build_index, (x, jnp.uint64(0)))
    return h

@partial(jax.jit, static_argnames=("bits",))
def hilbert_index_batch(points, bits):
    return jax.vmap(lambda p: hilbert_index_nd(p, bits))(points)

# -------------------------------
# BUILD GRID AND SORT
# -------------------------------

@partial(jax.jit, static_argnames=("bits",))
def build_hilbert_map(X, bits=8):
    mins = jnp.min(X, axis=0)
    maxs = jnp.max(X, axis=0)
    grid = jnp.floor((X-mins)/(maxs-mins+1e-9)*(2**bits-1)).astype(jnp.uint64)
    hilb = hilbert_index_batch(grid, bits)
    order = jnp.argsort(hilb)
    return X[order], order

# -------------------------------
# SLIDING WINDOW SMOOTHING
# -------------------------------

def hilbert_smooth(X_sorted, window=32, chunk_size=10000):
    N, D = X_sorted.shape
    pad = window//2
    padded = jnp.pad(X_sorted, ((pad,pad),(0,0)), mode='edge')
    smoothed_rows = []
    for i in range(0, N, chunk_size):
        chunk_indices = jnp.arange(i, min(i+chunk_size, N))
        def scan_fun(carry, idx):
            slice_i = lax.dynamic_slice(padded, (idx,0), (window,D))
            smoothed_row = jnp.mean(slice_i, axis=0)
            return carry, smoothed_row
        _, smoothed_chunk = lax.scan(scan_fun, None, chunk_indices)
        smoothed_rows.append(smoothed_chunk)
    return jnp.vstack(smoothed_rows)

# -------------------------------
# HILBERT ENSEMBLE PIPELINE
# -------------------------------

def hilbert_ensemble_map(X, reduced_dims=64, hilbert_bits=8, window=32, ensemble_size=4, key=jax.random.PRNGKey(0)):
    N, D = X.shape
    X_low = svd_reduce(X, reduced_dims, key)
    embeddings = []
    subkey = key
    for i in range(ensemble_size):
        subkey, perm_key = jax.random.split(subkey)
        perm = jax.random.permutation(perm_key, X_low.shape[1])
        X_perm = X_low[:, perm]
        X_sorted, order = build_hilbert_map(X_perm, hilbert_bits)
        X_smooth = hilbert_smooth(X_sorted, window)
        inverse_order = jnp.zeros_like(order)
        inverse_order = inverse_order.at[order].set(jnp.arange(N))
        embeddings.append(X_smooth[inverse_order])
    return jnp.mean(jnp.stack(embeddings, axis=0), axis=0)

# -------------------------------
# SYNTHETIC DATA (MILLIONS OF POINTS)
# -------------------------------

def generate_blobs(key, n_points=1_000_000, dims=10000, n_clusters=5, spread=0.5, chunk_size=50_000):
    points_per_cluster = n_points // n_clusters
    X_list = []
    subkey = key
    for i in range(n_clusters):
        subkey, sk = jax.random.split(subkey)
        center = jax.random.normal(sk, (dims,))*5.0
        cluster_points = []
        for j in range(0, points_per_cluster, chunk_size):
            batch_size = min(chunk_size, points_per_cluster-j)
            subkey, sk2 = jax.random.split(subkey)
            cluster_points.append(center + jax.random.normal(sk2,(batch_size,dims))*spread)
        X_list.append(jnp.vstack(cluster_points))
    return jnp.vstack(X_list)

# -------------------------------
# TEST
# -------------------------------

def run_test():
    key = jax.random.PRNGKey(0)
    N, D = 100000, 1000
    n_clusters = 5
    print(f"Generating {n_clusters} clusters with {N} points in {D}-dimensional space...")
    X = generate_blobs(key, N, D, n_clusters)
    print("Input shape:", X.shape)

    t0 = time.time()
    emb = hilbert_ensemble_map(X, reduced_dims=64, hilbert_bits=8, window=64, ensemble_size=6, key=key)
    jax.block_until_ready(emb)
    t1 = time.time()
    print("Ensemble embedding shape:", emb.shape)
    print("Time:", t1-t0,"seconds")

    # Optional visualization for first 2 dimensions (subsample for plotting)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        subsample = emb[:5000]
        plt.scatter(subsample[:,0], subsample[:,1], c=jnp.repeat(jnp.arange(n_clusters), 1000), cmap='tab10', s=2)
        plt.title("Hilbert Ensemble Map (1M+ points)")
        plt.show()
    except Exception:
        print("Matplotlib not available, skipping plot.")


def run_csv_test(filename=None):
    import pandas as pd
    if filename is None :
        run_test()
        return
    
    # Load CSV
    df = pd.read_csv(filename,sep='\t').iloc[:,1:]
    print(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include='number')
    X = jnp.array(df_numeric.values)
    print("Numeric data shape:", X.shape)

    # Run Hilbert-ensemble map
    key = jax.random.PRNGKey(123)
    t0 = time.time()
    embedding = hilbert_ensemble_map(
        X,
        reduced_dims=min(64, X.shape[1]),  # don't exceed original dims
        hilbert_bits=8,
        window=32,
        ensemble_size=4,
        key=key
    )
    jax.block_until_ready(embedding)
    t1 = time.time()
    print("Embedding shape:", embedding.shape)
    print("Time:", t1 - t0, "seconds")

    # Optional: plot first 2 dims
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(embedding[:,0], embedding[:,1], s=3, alpha=0.7)
        plt.title("Hilbert Ensemble Map (CSV data)")
        plt.show()
    except Exception:
        print("Matplotlib not available, skipping plot.")


"""DMap implementation."""
from ..base import BaseMap

class DMap(BaseMap):
    """Diffusion Map implementation."""
    
    def __init__(self, n_components=2, alpha=1.0, random_state=None):
        super().__init__(n_components, random_state)
        self.alpha = alpha
        
    def fit(self, X, y=None):
        # Your DMap implementation
        return self
    
    def transform(self, X):
        # Transform implementation
        pass

if __name__ == "__main__":
    run_test()
