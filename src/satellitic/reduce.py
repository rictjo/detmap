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
import time
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

import os
os.environ["JAX_ENABLE_X64"] = "True"


# ============================================================
# RANDOMIZED SVD
# ============================================================

@partial(jax.jit, static_argnames=("rank","oversample","n_iter"))
def randomized_svd(X, rank, oversample=8, n_iter=2, key=jax.random.PRNGKey(0)):

    N, D = X.shape
    l = rank + oversample

    Omega = jax.random.normal(key, (D, l))

    Y = X @ Omega

    for _ in range(n_iter):
        Y = X @ (X.T @ Y)

    Q, _ = jnp.linalg.qr(Y)

    B = Q.T @ X

    U_hat, S, Vt = jnp.linalg.svd(B, full_matrices=False)

    U = Q @ U_hat

    return U[:, :rank], S[:rank], Vt[:rank, :]


@partial(jax.jit, static_argnames=("rank",))
def svd_reduce(X, rank, key):

    U, S, V = randomized_svd(X, rank, key=key)

    return X @ V.T


# ============================================================
# MORTON INDEX
# ============================================================

@partial(jax.jit, static_argnames=("bits",))
def morton_index_nd(coords, bits):

    dims = coords.shape[0]
    coords = coords.astype(jnp.uint64)

    def body(i, h):

        b = bits - 1 - i

        digit = jnp.uint64(0)

        for d in range(dims):
            digit |= ((coords[d] >> b) & 1) << d

        h = (h << dims) | digit

        return h

    return lax.fori_loop(0, bits, body, jnp.uint64(0))


@partial(jax.jit, static_argnames=("bits",))
def morton_index_batch(points, bits):

    return jax.vmap(lambda p: morton_index_nd(p, bits))(points)


# ============================================================
# HILBERT INDEX
# ============================================================

@partial(jax.jit, static_argnames=("bits",))
def hilbert_index_nd(coords, bits):

    coords = coords.astype(jnp.uint64)
    n = coords.shape[0]
    x = coords

    def gray_step(i, x):

        bit = jnp.uint64(1) << (bits - 1 - i)

        def inner(j, x):

            cond = (x[j] & bit) != 0
            x = jnp.where(cond, x ^ (bit - 1), x)

            t = (x[0] ^ x[j]) & (bit - 1)

            x = x.at[0].set(x[0] ^ t)
            x = x.at[j].set(x[j] ^ t)

            return x

        return lax.fori_loop(1, n, inner, x)

    x = lax.fori_loop(0, bits, gray_step, x)

    def build_index(i, state):

        x, h = state
        b = bits - 1 - i

        digit = jnp.uint64(0)

        for d in range(n):
            digit |= ((x[d] >> b) & 1) << d

        h = (h << n) | digit

        return (x, h)

    _, h = lax.fori_loop(0, bits, build_index, (x, jnp.uint64(0)))

    return h


@partial(jax.jit, static_argnames=("bits",))
def hilbert_index_batch(points, bits):

    return jax.vmap(lambda p: hilbert_index_nd(p, bits))(points)


# ============================================================
# SPATIAL INDEX BUILD
# ============================================================

@partial(jax.jit, static_argnames=("bits",))
def build_spatial_index(X, bits):

    mins = jnp.min(X, axis=0)
    maxs = jnp.max(X, axis=0)

    grid = jnp.floor(
        (X - mins) / (maxs - mins + 1e-9) * (2**bits - 1)
    ).astype(jnp.uint64)

    morton = morton_index_batch(grid, bits)

    order = jnp.argsort(morton)

    grid_sorted = grid[order]

    hilbert = hilbert_index_batch(grid_sorted, bits)

    return grid_sorted, order, hilbert


# ============================================================
# APPROXIMATE KNN USING HILBERT WINDOW
# ============================================================

def hilbert_knn_search(X_sorted, query, k=10, window=64):

    dists = jnp.sum((X_sorted - query)**2, axis=1)

    idx = jnp.argsort(dists)

    return idx[:k]

# ============================================================
# HILBERT-BASED 2D MAP
# ============================================================

def hilbert_2d_map(X_low, bits=8, window=32, ensemble_size=4, key=jax.random.PRNGKey(0)):
    """
    Generate a 2D deterministic map from SVD-reduced data
    while preserving local neighborhoods.
    """

    N, D = X_low.shape
    embeddings = []

    subkey = key
    for _ in range(ensemble_size):
        # Optional: random permutation of dimensions
        subkey, sk = jax.random.split(subkey)
        perm = jax.random.permutation(sk, D)
        X_perm = X_low[:, perm]

        # Build Hilbert spatial index
        grid_sorted, order, hilbert = build_spatial_index(X_perm, bits)

        # Smooth positions along Hilbert curve
        pad = window // 2
        padded = jnp.pad(grid_sorted.astype(jnp.float32), ((pad,pad),(0,0)), mode='edge')

        smoothed = []
        for i in range(N):
            smoothed_row = jnp.mean(padded[i:i+window], axis=0)
            smoothed.append(smoothed_row)
        smoothed = jnp.stack(smoothed)

        # Project smoothed high-D coordinates to 2D via SVD
        _, _, Vt = jnp.linalg.svd(smoothed, full_matrices=False)
        emb2d = smoothed @ Vt.T[:, :2]

        # Reorder to original input order
        inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
        emb2d = emb2d[inv_order]

        embeddings.append(emb2d)

    # Average ensemble for stability
    return jnp.mean(jnp.stack(embeddings, axis=0), axis=0)


# ============================================================
# TEST PROGRAM
# ============================================================

def run_test(filename=None,k=5):

    key = jax.random.PRNGKey(0)

    N = 20000
    D = 1000
    reduced_dims = 32
    bits = 8

    if filename is None :
        print("Generating data...")
        X = jax.random.normal(key, (N, D))
    else:
        import pandas as pd
        X = pd.read_csv(filename,sep='\t').iloc[:,1:].values

    print("Data shape:", X.shape)

    # ------------------------------------------------
    # SVD dimensionality reduction
    # ------------------------------------------------

    print("Running randomized SVD reduction")

    t0 = time.time()

    X_low = svd_reduce(X, reduced_dims, key)

    jax.block_until_ready(X_low)

    t1 = time.time()

    print("Reduced shape:", X_low.shape)
    print("SVD time:", t1 - t0)

    # ------------------------------------------------
    # Build spatial index
    # ------------------------------------------------

    print("Building Morton/Hilbert spatial index")

    t0 = time.time()

    grid_sorted, order, hilbert = build_spatial_index(X_low, bits)

    jax.block_until_ready(grid_sorted)

    t1 = time.time()

    print("Index time:", t1 - t0)

    # ------------------------------------------------
    # Example nearest neighbor query
    # ------------------------------------------------

    query = X_low[0]

    print("Running approximate kNN search")

    t0 = time.time()

    neighbors = hilbert_knn_search(X_low, query, k=k)

    jax.block_until_ready(neighbors)

    t1 = time.time()

    print("Nearest neighbors indices:", neighbors)
    print("Query time:", t1 - t0)


    # ------------------------------------------------
    # Generate 2D map of the kNN structure
    # ------------------------------------------------

    print("Generating 2D Hilbert map...")

    t0 = time.time()
    map2d = hilbert_2d_map(X_low, bits=bits, window=32, ensemble_size=4, key=key)
    jax.block_until_ready(map2d)
    t1 = time.time()
    print("2D map shape:", map2d.shape)
    print("2D map time:", t1 - t0)

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.scatter(map2d[:,0], map2d[:,1], s=5, alpha=0.7)
        plt.title("2D Hilbert Map of SVD-reduced Data")
        plt.show()
    except Exception:
        print("Matplotlib not available, skipping plot.")

    print("Pipeline completed successfully")

# ============================================================


