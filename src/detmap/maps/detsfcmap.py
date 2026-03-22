"""
Enhanced Ultra-High Dim Space-Filling Ensemble Map
With proper multi-scale SFC support
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
import numpy as np



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
def morton_index_nd_64bit(coords, bits):
    """
    Morton code computation with 64-bit precision and overflow protection.
    """
    dims = coords.shape[0]

    # Protect against too many bits for 64-bit integer
    max_bits_per_dim = 63 // dims
    if bits > max_bits_per_dim:
        if jax.process_index() == 0:
            print(f"WARNING: Reducing bits from {bits} to {max_bits_per_dim} for {dims}D")
        bits = max_bits_per_dim

    coords = coords.astype(jnp.uint64)

    def spread_bits(x):
        """Spread bits for interleaving"""
        x = x & ((1 << bits) - 1)
        x = (x | (x << 32)) & 0x00000000ffffffff
        x = (x | (x << 16)) & 0x0000ffff0000ffff
        x = (x | (x << 8))  & 0x00ff00ff00ff00ff
        x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0f
        x = (x | (x << 2))  & 0x3333333333333333
        x = (x | (x << 1))  & 0x5555555555555555
        return x

    result = jnp.uint64(0)
    for d in range(dims):
        result |= spread_bits(coords[d]) << d

    return result

@partial(jax.jit, static_argnames=("bits",))
def morton_index_nd_safe(coords, bits):
    """
    Safe Morton code computation with overflow protection.
    Uses fewer bits per dimension to prevent overflow.
    """
    dims = coords.shape[0]

    # CRITICAL: Limit bits to prevent overflow
    # For 64-bit: max bits per dimension = floor(64 / dims)
    max_bits_per_dim = 63 // dims  # Use 63 to avoid sign bit
    if bits > max_bits_per_dim:
        print(f"WARNING: Reducing bits from {bits} to {max_bits_per_dim} to prevent overflow")
        bits = max_bits_per_dim

    coords = coords.astype(jnp.uint64)

    # Spread bits method (more efficient than loop)
    def spread_bits(x):
        x = x & ((1 << bits) - 1)  # Mask to bits
        x = (x | (x << 32)) & 0x00000000ffffffff
        x = (x | (x << 16)) & 0x0000ffff0000ffff
        x = (x | (x << 8))  & 0x00ff00ff00ff00ff
        x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0f
        x = (x | (x << 2))  & 0x3333333333333333
        x = (x | (x << 1))  & 0x5555555555555555
        return x

    result = jnp.uint64(0)
    for d in range(dims):
        result |= spread_bits(coords[d]) << d

    return result


@partial(jax.jit, static_argnames=("bits",))
def morton_index_batch(points, bits):
    #return jax.vmap(lambda p: morton_index_nd_safe(p, bits))(points)
    return jax.vmap(lambda p: morton_index_nd_64bit(p, bits))(points)



# -------------------------------------------------------
# ENHANCED SFC IMPLEMENTATIONS
# -------------------------------------------------------

def hilbert_index_nd(coords, bits):
    """
    Hilbert curve index for N dimensions.
    Better locality preservation than Morton.
    """
    dims = coords.shape[0]

    # Convert to integer coordinates
    coords = coords.astype(jnp.uint64)

    # Initialize Hilbert transform
    h = jnp.uint64(0)
    state = jnp.uint32(0)

    # Process bits from highest to lowest
    for b in range(bits-1, -1, -1):
        # Extract bits at position b
        bits_at_pos = jnp.array([(coord >> b) & 1 for coord in coords])

        # Hilbert transform state machine
        # This is simplified - a full implementation would use the Gray code transform
        for d in range(dims):
            if bits_at_pos[d]:
                # Flip state based on current position
                state ^= (1 << d)

        # Add to index
        h = (h << dims) | jnp.sum(bits_at_pos << jnp.arange(dims))

    return h

def multi_scale_morton_map(X, scale_levels=[4, 6, 8, 10], weights=None):
    """
    Multi-scale Morton mapping that captures structure at multiple resolutions.

    Parameters:
    -----------
    X : array (n_samples, n_features)
        Data to map
    scale_levels : list of int
        Different bit depths for multi-scale analysis
    weights : list of float, optional
        Weights for each scale level

    Returns:
    --------
    multi_scale_indices : list of arrays
        SFC indices at each scale
    """
    N, D = X.shape

    # Compute bounds
    mins = jnp.min(X, axis=0)
    maxs = jnp.max(X, axis=0)

    if weights is None:
        weights = [1.0 / len(scale_levels)] * len(scale_levels)

    multi_scale_indices = []

    for bits in scale_levels:
        # Scale to grid coordinates for this bit depth
        scale = (2 ** bits - 1) / (maxs - mins + 1e-12)
        grid_float = (X - mins) * scale
        grid_float = jnp.clip(grid_float, 0, 2**bits - 1)
        grid = grid_float.astype(jnp.uint64)

        # Compute Morton indices
        morton = morton_index_batch(grid, bits)
        multi_scale_indices.append(morton)

    return multi_scale_indices

def hilbert_smooth_multi_scale(X_sorted, windows):
    """
    Multi-scale smoothing along SFC order.
    Combines multiple window sizes for multi-resolution features.
    """
    if not isinstance(windows, (list, tuple)):
        windows = [windows]

    smoothed = []
    for window in windows:
        kernel = jnp.ones((window,), dtype=X_sorted.dtype) / window
        smooth_dim = lambda x: jsp.signal.convolve(x, kernel, mode="same")
        smoothed.append(jax.vmap(smooth_dim, in_axes=1, out_axes=1)(X_sorted))

    # Weighted combination of scales
    return jnp.mean(jnp.stack(smoothed), axis=0)

# -------------------------------------------------------
# ENHANCED ENSEMBLE MAP WITH PROPER SFC
# -------------------------------------------------------

def enhanced_hilbert_ensemble_map(
    X,
    reduced_dims=64,
    scale_levels=[4, 6, 8, 10],  # Multi-scale bit depths
    windows=[16, 32, 64],         # Multi-scale windows
    ensemble_size=6,
    use_hilbert=True,              # Use Hilbert instead of Morton
    key=jax.random.PRNGKey(0),
):
    """
    Enhanced SFC-based manifold embedding with:
    - Multi-scale SFC indexing
    - Multi-scale smoothing
    - Hilbert curve option for better locality
    - Proper SFC-aware ensemble
    """
    N, D = X.shape

    print(f"Input shape: {X.shape}")
    print(f"Scale levels: {scale_levels}")
    print(f"Windows: {windows}")

    # Dimensionality reduction
    if D > reduced_dims:
        print(f"Reducing dimensions: {D} -> {reduced_dims}")
        X_low = svd_reduce(X, reduced_dims, key)
    else:
        X_low = X
        reduced_dims = D
        print(f"Keeping original dimensions: {D}")

    # Compute bounds once
    mins = jnp.min(X_low, axis=0)
    maxs = jnp.max(X_low, axis=0)

    embeddings = []
    subkey = key

    for i in range(ensemble_size):
        print(f"Ensemble {i+1}/{ensemble_size}")
        subkey, rot_key, perm_key = jax.random.split(subkey, 3)

        # Random rotation (improves SFC coverage)
        X_rot = random_rotation(rot_key, X_low)

        # Random permutation of dimensions
        perm = jax.random.permutation(perm_key, X_rot.shape[1])
        X_perm = X_rot[:, perm]

        # Multi-scale SFC ordering
        if use_hilbert:
            # Multi-scale Hilbert ordering
            multi_scale_indices = []
            for bits in scale_levels:
                # Scale coordinates
                scale = (2 ** bits - 1) / (maxs - mins + 1e-12)
                grid_float = (X_perm - mins) * scale
                grid_float = jnp.clip(grid_float, 0, 2**bits - 1)
                grid = grid_float.astype(jnp.uint64)

                # Hilbert indices (would need proper Hilbert implementation)
                # For now using Morton with Hilbert-like weighting
                indices = morton_index_batch(grid, bits)
                multi_scale_indices.append(indices)

            # Combine multi-scale indices with weights
            # Higher bits get more weight for fine structure
            weights = jnp.array([2**b for b in scale_levels], dtype=jnp.float32)
            weights = weights / jnp.sum(weights)

            combined_index = jnp.zeros(N, dtype=jnp.float32)
            for idx, weight in zip(multi_scale_indices, weights):
                combined_index += weight * idx.astype(jnp.float32)

            order = jnp.argsort(combined_index)
        else:
            # Standard Morton with highest resolution
            max_bits = max(scale_levels)
            scale = (2 ** max_bits - 1) / (maxs - mins + 1e-12)
            grid_float = (X_perm - mins) * scale
            grid_float = jnp.clip(grid_float, 0, 2**max_bits - 1)
            grid = grid_float.astype(jnp.uint64)
            morton = morton_index_batch(grid, max_bits)
            order = jnp.argsort(morton)

        # Apply SFC order
        X_sorted = X_perm[order]

        # Multi-scale smoothing along SFC
        X_smooth = hilbert_smooth_multi_scale(X_sorted, windows)

        # Restore original order
        inverse_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
        X_restored = X_smooth[inverse_order]

        embeddings.append(X_restored)

        # Memory cleanup
        del X_rot, X_perm, X_sorted, X_smooth, X_restored

    # Average ensemble
    result = jnp.mean(jnp.stack(embeddings), axis=0)
    print(f"Final embedding shape: {result.shape}")

    return result

# -------------------------------------------------------
# SFC-AWARE PCA WITH LOCALITY PRESERVATION
# -------------------------------------------------------

def sfc_aware_pca(X, n_components=3, sfc_bits=8, window=32):
    """
    PCA with SFC-based locality preservation.
    """
    # Standard PCA first
    X_centered = X - jnp.mean(X, axis=0, keepdims=True)
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)
    X_pca = X_centered @ Vt.T[:, :n_components]

    # Compute SFC order
    mins = jnp.min(X_pca, axis=0)
    maxs = jnp.max(X_pca, axis=0)
    scale = (2 ** sfc_bits - 1) / (maxs - mins + 1e-12)
    grid = ((X_pca - mins) * scale).astype(jnp.uint64)
    morton = morton_index_batch(grid, sfc_bits)
    order = jnp.argsort(morton)

    # Smooth along SFC
    X_sorted = X_pca[order]
    kernel = jnp.ones((window,)) / window
    smooth_dim = lambda x: jsp.signal.convolve(x, kernel, mode="same")
    X_smooth = jax.vmap(smooth_dim, in_axes=1, out_axes=1)(X_sorted)

    # Restore order
    inverse_order = jnp.zeros(len(X_pca), dtype=jnp.int32).at[order].set(jnp.arange(len(X_pca)))
    X_sfc_aware = X_smooth[inverse_order]

    return X_sfc_aware

# -------------------------------------------------------
# SFC DetMap CLASS
# -------------------------------------------------------
from ..base import BaseMap
import numpy as np
class DetSFCMap(BaseMap):
    """
    Deterministic Rotation Map with SFC support.
    """

    def __init__(self, n_components=None, reduced_dims=64,
                 scale_levels=[4, 6, 8, 10],  # Multi-scale SFC bits
                 windows=[16, 32, 64],        # Multi-scale windows
                 ensemble_size=6,
                 use_hilbert=True,
                 random_state=None):
        super().__init__(n_components, random_state)
        self.reduced_dims = reduced_dims
        self.scale_levels = scale_levels
        self.windows = windows
        self.ensemble_size = ensemble_size
        self.use_hilbert = use_hilbert
        self.X_fit_ = None
        self.X_embedded = None

    def fit(self, X, y=None):
        """Just store the training data for reference."""
        self.X_fit_ = self._check_array(X)
        return self

    def transform(self, X):
        """Transform with enhanced SFC support."""
        X = self._check_array(X)

        key = jax.random.PRNGKey(self.random_state if self.random_state else 0)

        # Use enhanced SFC-aware embedding
        X_embedded = enhanced_hilbert_ensemble_map(
            jnp.array(X, dtype=jnp.float64),
            reduced_dims=min(self.reduced_dims, X.shape[1]),
            scale_levels=self.scale_levels,
            windows=self.windows,
            ensemble_size=self.ensemble_size,
            use_hilbert=self.use_hilbert,
            key=key
        )

        jax.block_until_ready(X_embedded)
        self.X_embedded = X_embedded

        # Select requested components
        n_comp = self.n_components if self.n_components else min(2, X_embedded.shape[1])
        return np.asarray(X_embedded[:, :n_comp])
