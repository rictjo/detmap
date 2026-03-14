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
bUseJax = False
try :
        import jax
        bUseJax = True
except ImportError :
        bUseJax = False
except OSError:
        bUseJax = False

import jax.numpy as jnp
import numpy as np
def strings_find(a, sub, start=0, end=None):
    """
    En ren Python-version av numpy.strings.find.
    
    a   : en itererbar av strängar
    sub : substring att söka efter
    start, end : valfritt sökområde
    """
    results = []
    for s in a:
        # Python slice hanteras direkt av str.find
        idx = str(s).find(sub, start, end)
        results.append(idx)
    return np.array(results)

if bUseJax :
    import jax
    import jax.numpy as jnp
    from jax import lax
    #
    # ============================================================
    # Morton curve methods
    @jax.jit
    def _split_by_3bits(x):
        x &= 0x1fffff
        x = (x | (x << 32)) & 0x1f00000000ffff
        x = (x | (x << 16)) & 0x1f0000ff0000ff
        x = (x | (x << 8))  & 0x100f00f00f00f00f
        x = (x | (x << 4))  & 0x10c30c30c30c30c3
        x = (x | (x << 2))  & 0x1249249249249249
        return x

    @jax.jit
    def morton3D(x, y, z):
        return (_split_by_3bits(x) |
               (_split_by_3bits(y) << 1) |
               (_split_by_3bits(z) << 2))

    @jax.jit
    def compute_cells(r, cell_size):
        cells = jnp.floor(r / cell_size).astype(jnp.int32)
        # shift so negative coordinates work
        cells = cells - jnp.min(cells, axis=0)
        return cells

    @jax.jit
    def compute_morton_codes(r, cell_size):
        cells = compute_cells(r, cell_size)
        codes = morton3D(
            cells[:,0].astype(jnp.uint64),
            cells[:,1].astype(jnp.uint64),
            cells[:,2].astype(jnp.uint64)
        )
        return cells, codes
    #
    # ============================================================
    # Hilbert curve methods
    @jax.jit
    def hilbert3D(x, y, z, bits=21):
        """
    Fast 3D Hilbert index.

    x,y,z : uint64 coordinates
    bits  : number of bits per dimension (<=21)

    returns uint64 Hilbert index
        """
        x = x.astype(jnp.uint64)
        y = y.astype(jnp.uint64)
        z = z.astype(jnp.uint64)
        mask = jnp.uint64(1) << (bits - 1)

        def body(i, state):
            x, y, z, h, mask = state
            xi = (x & mask) > 0
            yi = (y & mask) > 0
            zi = (z & mask) > 0

            digit = (
                (xi.astype(jnp.uint64) << 2) |
                (yi.astype(jnp.uint64) << 1) |
                zi.astype(jnp.uint64)
            )
            h = (h << 3) | digit

            # rotation / reflection step
            swap_xy = (~zi) & (xi ^ yi)
            swap_xz = (~yi) & (xi ^ zi)

            x2 = jnp.where(swap_xy, y, x)
            y2 = jnp.where(swap_xy, x, y)
            x3 = jnp.where(swap_xz, z, x2)
            z2 = jnp.where(swap_xz, x2, z)
            mask = mask >> 1
            return (x3, y2, z2, h, mask)

        x, y, z, h, mask = lax.fori_loop(
            0,
            bits,
            body,
            (x, y, z, jnp.uint64(0), mask)
        )
        return h
   
    @jax.jit
    def hilbert3D_vec(x, y, z, bits=21):
        return jax.vmap(lambda a, b, c: hilbert3D(a, b, c, bits))(x, y, z)

    @jax.jit
    def compute_hilbert_codes(r, cell_size, bits=21):
        cells = jnp.floor(r / cell_size).astype(jnp.int32)
        # shift so negatives work
        cells = cells - jnp.min(cells, axis=0)
        x = cells[:,0].astype(jnp.uint64)
        y = cells[:,1].astype(jnp.uint64)
        z = cells[:,2].astype(jnp.uint64)
        codes = hilbert3D_vec(x, y, z, bits)
        return cells, codes

    # ============================================================
    # Build cell start/end lookup
    #
    @jax.jit
    def build_cell_table(codes):
        order = jnp.argsort(codes)
        codes_sorted = codes[order]
        unique, start = jnp.unique(codes_sorted, return_index=True)
        end = jnp.concatenate([start[1:], jnp.array([codes_sorted.shape[0]])])
        return order, codes_sorted, unique, start, end

else :
    def _split_by_3bits(x):
        x &= 0x1fffff
        x = (x | (x << 32)) & 0x1f00000000ffff
        x = (x | (x << 16)) & 0x1f0000ff0000ff
        x = (x | (x << 8))  & 0x100f00f00f00f00f
        x = (x | (x << 4))  & 0x10c30c30c30c30c3
        x = (x | (x << 2))  & 0x1249249249249249
        return x
    
    def morton3D(x, y, z):
        return (_split_by_3bits(x) |
               (_split_by_3bits(y) << 1) |
               (_split_by_3bits(z) << 2))

    def compute_cells(r, cell_size):
        cells = np.floor(r / cell_size).astype(jnp.int32)
        # shift so negative coordinates work
        cells = cells - np.min(cells, axis=0)
        return cells

    def compute_morton_codes(r, cell_size):
        cells = compute_cells(r, cell_size)
        codes = morton3D(
            cells[:,0].astype(np.uint64),
            cells[:,1].astype(np.uint64),
            cells[:,2].astype(np.uint64)
        )
        return cells, codes

    # ============================================================
    # Build cell start/end lookup
    #
    def build_cell_table(codes):
        order = np.argsort(codes)
        codes_sorted = codes[order]
        unique, start = np.unique(codes_sorted, return_index=True)
        end = np.concatenate([start[1:], jnp.array([codes_sorted.shape[0]])])
        return order, codes_sorted, unique, start, end

def rankdata_jax(a, method='average'):
    """
    Rank the data, similar to scipy.stats.rankdata.

    Args:
        a: 1D array-like
        method: how to handle ties; only 'average' is implemented

    Returns:
        ranks: 1D array of ranks (float), starting at 1
    """
    a = jnp.array(a)
    sort_idx = jnp.argsort(a)
    sorted_a = a[sort_idx]
    
    # Compute ranks
    ranks = jnp.arange(1, len(a)+1, dtype=jnp.float32)
    
    if method == 'average':
        # Handle ties by averaging their ranks
        diff = jnp.diff(sorted_a, prepend=sorted_a[0]-1)
        tie_groups = jnp.cumsum(diff != 0)  # group ID for each tie
        # Compute sum of ranks per group
        group_sum = jnp.zeros(tie_groups.max()+1)
        group_count = jnp.zeros(tie_groups.max()+1)
        group_sum = group_sum.at[tie_groups].add(ranks)
        group_count = group_count.at[tie_groups].add(1)
        avg_ranks = group_sum / group_count
        ranks = avg_ranks[tie_groups]
    
    else:
        raise NotImplementedError(f"Only method='average' is implemented, got {method}")
    
    # Undo sorting to original order
    inv_sort_idx = jnp.argsort(sort_idx)
    return ranks[inv_sort_idx]
  
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
    try:
        from sklearn.datasets import make_blobs
        from sklearn.decomposition import PCA
    except:
        print('sklearn not present')
        return False
       
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


def local_pca_jax(df, ndims=None, random_key=jax.random.PRNGKey(42)):
    """
    Local PCA wrapper using JAX randomized PCA.

    Args:
        df: pandas DataFrame (n_samples x n_features)
        ndims: number of components to keep (default: all)
        random_key: JAX random key for reproducibility

    Returns:
        scores: projected data (n_samples x ndims)
        weights: principal components (n_features x ndims)
        row_index: original df index
        col_names: original df columns
    """
    X = jnp.array(df.values)
    n_components = ndims if ndims is not None else min(X.shape)
    
    # Run randomized PCA
    X_pca, components, _ = randomized_pca_jax(X, n_components=n_components, random_key=random_key)

    # Note: sklearn returns components as (n_components x n_features),
    # but your wrapper expects weights as (n_features x n_components)
    weights = jnp.array(components).T

    # Return as tuple matching the original local_pca
    return X_pca, weights, df.index, df.columns




if __name__ == '__main__':
    a = ["hello world", "test string", "abcabc", "no match"]
    print ( strings_find(a, "abc") )

    a = np.array(["NumPy is a Python library"])
    print( strings_find(a, "Python") )

    print( 'spam, spam, spam'.find('sp', 1, None) )
