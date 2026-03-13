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
Deterministic manifold embedding using:

• Random projection
• Multi-SFC ensembles
• Barnes–Hut style hierarchical smoothing
• Local PCA refinement

Runs on JAX and scales approximately O(N).
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial


# --------------------------------------------------
# Random projection
# --------------------------------------------------

def random_projection(X, k, key):

    D = X.shape[1]

    R = jax.random.normal(key, (D, k)) / jnp.sqrt(k)

    return X @ R


# --------------------------------------------------
# Morton codes
# --------------------------------------------------

def morton_codes(points, bits=16):

    p = points - jnp.min(points, axis=0)
    p = p / (jnp.max(p, axis=0) + 1e-9)

    p = (p * (2**bits - 1)).astype(jnp.uint32)

    x = p[:,0]
    y = p[:,1]

    def part1by1(n):

        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    return part1by1(x) | (part1by1(y) << 1)


# --------------------------------------------------
# Hierarchical smoothing
# --------------------------------------------------
from functools import partial

@partial(jit, static_argnums=(1,))
def hierarchical_smooth(X, levels):

    N, D = X.shape

    result = jnp.zeros_like(X)
    weight = jnp.zeros((N,1))

    for l in range(levels):

        block = 2 ** l

        idx = jnp.arange(N) // block
        idx = jnp.clip(idx, 0, N-1)

        centers = jax.ops.segment_sum(X, idx, N)
        counts  = jax.ops.segment_sum(jnp.ones((N,1)), idx, N)

        centers = centers / (counts + 1e-9)

        w = 1.0 / block

        result += w * centers[idx]
        weight += w

    return result / weight


# --------------------------------------------------
# Local PCA refinement
# --------------------------------------------------

@jit
def local_pca(points):

    mean = jnp.mean(points, axis=0)
    Xc = points - mean

    cov = Xc.T @ Xc

    vals, vecs = jnp.linalg.eigh(cov)

    top2 = vecs[:, -2:]

    return Xc @ top2


def pca_windows(X, window=32):

    N = X.shape[0]

    out = []

    for i in range(0, N, window):

        block = X[i:i+window]

        if block.shape[0] < 3:
            continue

        out.append(local_pca(block))

    return jnp.concatenate(out, axis=0)


# --------------------------------------------------
# Single SFC embedding
# --------------------------------------------------

def single_sfc_embedding(X, key,
                         proj_dim,
                         levels):

    Xp = random_projection(X, proj_dim, key)

    codes = morton_codes(Xp[:, :2])

    order = jnp.argsort(codes)

    Xs = Xp[order]

    Xs = hierarchical_smooth(Xs, levels)

    Xs = pca_windows(Xs)

    inv = jnp.empty_like(order)
    inv = inv.at[order].set(jnp.arange(order.shape[0]))

    return Xs[inv]


# --------------------------------------------------
# Ensemble embedding
# --------------------------------------------------

def ensemble_embedding(X,
                       proj_dim=64,
                       levels=6,
                       ensemble=4,
                       seed=0):

    key = jax.random.PRNGKey(seed)

    Ys = []

    for i in range(ensemble):

        key, sub = jax.random.split(key)

        Y = single_sfc_embedding(
            X,
            sub,
            proj_dim,
            levels
        )

        Ys.append(Y)

    Y = jnp.mean(jnp.stack(Ys), axis=0)

    return Y


# --------------------------------------------------
# Data loader
# --------------------------------------------------

def load_dataframe_test():

    path = Path("analytes.csv")

    if path.exists():

        print("Loading analytes.csv")

        df = pd.read_csv(path,sep='\t')

        numeric = df.select_dtypes(include=[np.number])
        print(numeric)
        return jnp.array(numeric.values)

    else:

        print("Generating synthetic dataset")

        n = 6000
        d = 50

        rng = np.random.default_rng(0)

        c1 = rng.normal(0,1,(n//2,d))
        c2 = rng.normal(4,1,(n//2,d))

        X = np.vstack([c1,c2])

        return jnp.array(X)


# --------------------------------------------------
# Main test
# --------------------------------------------------

def main():

    X = load_dataframe_test()

    Y = ensemble_embedding(
        X,
        proj_dim=64,
        levels=7,
        ensemble=4
    )

    Y = np.array(Y)

    print("Embedding shape:", Y.shape)

    plt.figure(figsize=(8,8))
    plt.scatter(Y[:,0], Y[:,1], s=3)
    plt.title("Deterministic SFC Ensemble Embedding")
    plt.show()


if __name__ == "__main__":
    main()
