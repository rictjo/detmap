"""
bitmap.py - Safe bit-interleaved hybrid curve embedding with overflow protection
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

# ============================================================
# SAFE BIT OPERATIONS WITH OVERFLOW PROTECTION
# ============================================================

@partial(jax.jit, static_argnames=("bits", "dims"))
def morton_index_nd_safe(coords, bits, dims):
    """
    Safe Morton index computation with overflow protection.
    Uses 32-bit intermediates to prevent overflow.
    """
    # Limit bits to prevent overflow (31 bits total for safety)
    max_safe_bits = 31 // dims
    actual_bits = min(bits, max_safe_bits)

    # Convert to 32-bit first to prevent overflow
    coords = coords.astype(jnp.uint32)

    def spread_bits(x):
        # Mask to safe bits
        x = x & ((1 << actual_bits) - 1)

        # All operations in 32-bit space
        x = (x | (x << 16)) & 0x0000ffff
        x = (x | (x << 8)) & 0x00ff00ff
        x = (x | (x << 4)) & 0x0f0f0f0f
        x = (x | (x << 2)) & 0x33333333
        x = (x | (x << 1)) & 0x55555555
        return x.astype(jnp.uint64)  # Convert back at the end

    result = jnp.uint64(0)
    for d in range(dims):
        result |= spread_bits(coords[d]) << d

    return result


@partial(jax.jit, static_argnames=("bits", "dims"))
def hilbert_index_nd_safe(coords, bits, dims):
    """
    Safe Hilbert index computation with overflow protection.
    Uses 32-bit intermediates.
    """
    # Limit bits to prevent overflow
    max_safe_bits = 31 // dims
    actual_bits = min(bits, max_safe_bits)

    # Convert to 32-bit first
    coords = coords.astype(jnp.uint32)
    x = coords.astype(jnp.uint64)  # Convert back for shifts

    def gray_step(i, x):
        bit = jnp.uint64(1) << (actual_bits - 1 - i)

        def inner(j, x):
            cond = (x[j] & bit) != 0
            x = jnp.where(cond, x ^ (bit - 1), x)
            t = (x[0] ^ x[j]) & (bit - 1)
            x = x.at[0].set(x[0] ^ t)
            x = x.at[j].set(x[j] ^ t)
            return x

        return lax.fori_loop(1, dims, inner, x)

    x = lax.fori_loop(0, actual_bits, gray_step, x)

    def build_index(i, state):
        x, h = state
        b = actual_bits - 1 - i
        digit = jnp.uint64(0)
        for d in range(dims):
            digit |= ((x[d] >> b) & 1) << d
        h = (h << dims) | digit
        return (x, h)

    _, h = lax.fori_loop(0, actual_bits, build_index, (x, jnp.uint64(0)))
    return h


@partial(jax.jit, static_argnames=("bits", "dims"))
def weighted_bit_interleaved_safe(coords, bits, dims):
    """
    Weighted bit interleaving with overflow protection.
    """
    morton = morton_index_nd_safe(coords, bits, dims)
    hilbert = hilbert_index_nd_safe(coords, bits, dims)

    # Weighted combination - more weight to Hilbert for better locality
    morton_weight = 0.3
    hilbert_weight = 0.7

    return (morton * morton_weight + hilbert * hilbert_weight).astype(jnp.uint64)


def compute_safe_bits(dims, requested_bits=6):
    """
    Compute safe number of bits per dimension given dimensions.
    """
    max_bits = 31 // max(1, dims)
    return min(requested_bits, max(1, max_bits))


# ============================================================
# STATIC SMOOTHING (NO DYNAMIC SLICES)
# ============================================================

def gaussian_smooth_static(X_sorted, window):
    """
    Static Gaussian smoothing using convolution (JAX-compatible).
    No dynamic slices - uses jax.scipy.signal.convolve.
    """
    from jax.scipy.signal import convolve

    N = X_sorted.shape[0]
    sigma = max(1.0, window / 4.0)
    positions = jnp.arange(window) - window // 2
    weights = jnp.exp(-(positions ** 2) / (2 * sigma ** 2))
    weights = weights / (jnp.sum(weights) + 1e-9)

    def smooth_dim(x):
        # Pad with edge values
        pad = window // 2
        x_padded = jnp.pad(x, (pad, pad), mode='edge')
        # Convolve with weights
        result = convolve(x_padded, weights, mode='valid')
        return result[:N]

    return jax.vmap(smooth_dim, in_axes=1, out_axes=1)(X_sorted)


def random_rotation_safe(key, X):
    """
    Safe random orthogonal rotation.
    """
    D = X.shape[1]
    M = jax.random.normal(key, (D, D), dtype=X.dtype)
    Q, _ = jnp.linalg.qr(M)
    return X @ Q


def randomized_svd_safe(X, rank, key):
    """
    Safe randomized SVD with memory efficiency.
    """
    N, D = X.shape
    rank = min(rank, N, D)

    # Center the data
    X_centered = X - jnp.mean(X, axis=0)

    # Use randomized algorithm
    oversample = min(8, rank)
    l = rank + oversample

    Omega = jax.random.normal(key, (D, l), dtype=X.dtype)

    # Power iterations
    Y = X_centered @ Omega

    for _ in range(2):
        Y = X_centered @ (X_centered.T @ Y)

    Q, _ = jnp.linalg.qr(Y)

    # Reduce to smaller matrix
    B = Q.T @ X_centered

    # SVD on smaller matrix
    U_hat, S, Vt = jnp.linalg.svd(B, full_matrices=False)

    # Project back
    U = Q @ U_hat

    # Return scores
    return U[:, :rank] * S[:rank][None, :]


# ============================================================
# CLUSTER DETECTION FROM CURVE
# ============================================================

def detect_clusters_from_curve_safe(indices, threshold=2.5, min_cluster_size=3):
    """
    Detect clusters from sorted curve indices using static operations.
    """
    # Sort indices
    order = jnp.argsort(indices)
    sorted_indices = indices[order]

    # Compute gaps
    gaps = sorted_indices[1:] - sorted_indices[:-1]
    gaps = gaps.astype(jnp.float64)

    # Use global median for threshold (simpler and JIT-friendly)
    median_gap = jnp.median(gaps)
    gap_threshold = median_gap * threshold

    # Find cluster boundaries where gaps exceed threshold
    boundaries = jnp.where(gaps > gap_threshold)[0] + 1

    # Build cluster labels
    labels = jnp.zeros(len(indices), dtype=jnp.int32)

    # Handle no boundaries case
    def assign_clusters(carry, boundary):
        labels, start, cluster_id = carry
        end = boundary
        if end - start >= min_cluster_size:
            labels = labels.at[order[start:end]].set(cluster_id)
            return (labels, end, cluster_id + 1)
        return (labels, end, cluster_id)

    # Process boundaries
    if len(boundaries) > 0:
        # First segment
        first_end = boundaries[0]
        if first_end >= min_cluster_size:
            labels = labels.at[order[:first_end]].set(0)
            start = first_end
            cluster_id = 1
        else:
            start = first_end
            cluster_id = 0

        # Remaining segments
        for i in range(1, len(boundaries)):
            end = boundaries[i]
            if end - start >= min_cluster_size:
                labels = labels.at[order[start:end]].set(cluster_id)
                cluster_id += 1
            start = end

        # Last segment
        if len(indices) - start >= min_cluster_size:
            labels = labels.at[order[start:]].set(cluster_id)

    return labels


# ============================================================
# OPTIMAL HYBRID MAP CLASS
# ============================================================

class OptimalHybridMap:
    """
    Safe optimal hybrid curve embedding with overflow protection.
    """

    def __init__(self,
                 n_components=2,
                 reduced_dims=32,
                 bits=6,
                 window=16,
                 ensemble_size=3,
                 random_state=42):
        self.n_components = n_components
        self.reduced_dims = reduced_dims
        self.bits = bits
        self.window = window
        self.ensemble_size = ensemble_size
        self.random_state = random_state
        self.embedded_ = None
        self.labels_ = None
        self.safe_bits_ = None

    def fit_transform(self, X):
        """
        Fit and transform the data.

        Args:
            X: Input data (samples, features)

        Returns:
            Embedded coordinates (samples, n_components)
        """
        key = jax.random.PRNGKey(self.random_state)

        # Convert to JAX array
        X_jax = jnp.array(X, dtype=jnp.float32)

        # Step 1: Dimensionality reduction
        if X.shape[1] > self.reduced_dims:
            X_low = randomized_svd_safe(X_jax, self.reduced_dims, key)
        else:
            X_low = X_jax
            self.reduced_dims = X.shape[1]

        # Step 2: Compute safe bits for reduced dimensions
        self.safe_bits_ = compute_safe_bits(self.reduced_dims, self.bits)

        # Step 3: Normalize
        mins = jnp.min(X_low, axis=0)
        maxs = jnp.max(X_low, axis=0)

        # Step 4: Weighted bit-interleaved embedding
        X_embedded, indices = self._weighted_bit_interleaved_embedding(
            X_low, mins, maxs, key
        )

        # Step 5: Cluster detection
        self.labels_ = detect_clusters_from_curve_safe(indices)
        self.embedded_ = X_embedded

        # Return requested components
        return np.asarray(self.embedded_[:, :self.n_components])

    def _weighted_bit_interleaved_embedding(self, X, mins, maxs, key):
        """
        Weighted bit-interleaved embedding with safe operations.
        """
        N, D = X.shape
        bits = self.safe_bits_
        window = self.window

        # Safe grid computation
        scale = (2.0**bits - 1.0) / (maxs - mins + 1e-9)
        grid = jnp.floor((X - mins) * scale).astype(jnp.uint64)

        # Clip to prevent overflow
        max_val = 2**bits - 1
        grid = jnp.clip(grid, 0, max_val)

        embeddings = []
        all_indices = []
        subkey = key

        for i in range(self.ensemble_size):
            subkey, rot_key, perm_key = jax.random.split(subkey, 3)

            # Random rotation and permutation
            X_rot = random_rotation_safe(rot_key, X)
            perm = jax.random.permutation(perm_key, D)
            X_perm = X_rot[:, perm]

            # Recompute grid for rotated data
            mins_rot = jnp.min(X_perm, axis=0)
            maxs_rot = jnp.max(X_perm, axis=0)
            scale_rot = (2.0**bits - 1.0) / (maxs_rot - mins_rot + 1e-9)
            grid_rot = jnp.floor((X_perm - mins_rot) * scale_rot).astype(jnp.uint64)
            grid_rot = jnp.clip(grid_rot, 0, max_val)

            # Compute weighted indices
            indices = jax.vmap(
                lambda p: weighted_bit_interleaved_safe(p, bits, D)
            )(grid_rot)

            # Sort and smooth
            order = jnp.argsort(indices)
            X_sorted = X_perm[order]
            X_smooth = gaussian_smooth_static(X_sorted, window)

            # Restore original order
            inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
            X_restored = X_smooth[inv_order]

            embeddings.append(X_restored)

            if i == 0:
                all_indices.append(indices)

        return jnp.mean(jnp.stack(embeddings), axis=0), all_indices[0]

    def get_embedding(self):
        """Return the full embedding."""
        return self.embedded_

    def get_labels(self):
        """Return cluster labels."""
        return self.labels_


# ============================================================
# BACKWARD COMPATIBLE WRAPPER
# ============================================================

class BitInterleavedClusterMap(OptimalHybridMap):
    """Backward-compatible wrapper."""
    pass


# ============================================================
# SIMPLE API FUNCTION
# ============================================================

def detmap_embedding(X, n_components=2, reduced_dims=32, bits=6,
                     window=16, ensemble_size=3, random_state=42):
    """
    Simple function interface for DetMap embedding.
    """
    mapper = OptimalHybridMap(
        n_components=n_components,
        reduced_dims=reduced_dims,
        bits=bits,
        window=window,
        ensemble_size=ensemble_size,
        random_state=random_state
    )

    embedded = mapper.fit_transform(X)
    labels = mapper.get_labels()

    return embedded, labels



"""
bitmap_optimized.py - Enhanced hybrid map with statistical optimization for better cluster separation
"""

# ============================================================
# STATISTICAL METRICS FOR CLUSTER SEPARATION
# ============================================================

@partial(jax.jit, static_argnames=("power",))
def compute_gap_statistics(gaps, power=1.0):
    """
    Compute statistical metrics on gaps to measure cluster separation.
    Returns metrics that can be used as optimization objectives.
    """
    if power != 1.0:
        gaps = gaps ** power

    # Variance - higher variance means more separated clusters
    variance = jnp.var(gaps)

    # Kurtosis - higher kurtosis means more extreme gaps (better separation)
    mean_gap = jnp.mean(gaps)
    centered = gaps - mean_gap
    kurtosis = jnp.mean(centered ** 4) / (jnp.mean(centered ** 2) ** 2 + 1e-9)

    # Inter-quartile range - captures separation magnitude
    q75, q25 = jnp.percentile(gaps, 75), jnp.percentile(gaps, 25)
    iqr = q75 - q25

    # Max gap ratio - ratio of largest gap to median
    max_ratio = jnp.max(gaps) / (jnp.median(gaps) + 1e-9)

    # Combine metrics into a single separation score
    # Higher score = better separation
    separation_score = (
        variance * 0.3 +
        (kurtosis - 3) * 0.2 +  # Excess kurtosis (0 = normal distribution)
        iqr * 0.3 +
        max_ratio * 0.2
    )

    return {
        'variance': variance,
        'kurtosis': kurtosis,
        'iqr': iqr,
        'max_ratio': max_ratio,
        'separation_score': separation_score
    }


@partial(jax.jit, static_argnames=("n_bins",))
def compute_bin_statistics(indices, n_bins=20):
    """
    Compute bin-based statistics to measure cluster separation.
    Good clusters should create a multimodal distribution.
    """
    # Create bins along the curve
    min_idx = jnp.min(indices)
    max_idx = jnp.max(indices)
    bin_edges = jnp.linspace(min_idx, max_idx, n_bins + 1)

    # Count points per bin
    bin_counts = jnp.histogram(indices, bins=bin_edges)[0]

    # Metrics for multimodal distribution
    # Shannon entropy - lower entropy means more concentrated clusters
    probs = bin_counts / (jnp.sum(bin_counts) + 1e-9)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-9))

    # Number of bins with significant counts
    significant_bins = jnp.sum(bin_counts > jnp.mean(bin_counts))

    # Peak-to-average ratio
    peak_ratio = jnp.max(bin_counts) / (jnp.mean(bin_counts) + 1e-9)

    return {
        'entropy': entropy,
        'significant_bins': significant_bins,
        'peak_ratio': peak_ratio,
        'multimodality_score': peak_ratio / (entropy + 1e-9)
    }


# ============================================================
# OPTIMIZED HYBRID INDEX WITH ADAPTIVE WEIGHTS
# ============================================================

@partial(jax.jit, static_argnames=("bits", "dims", "morton_weight", "hilbert_weight", "exaggeration"))
def weighted_bit_interleaved_adaptive(coords, bits, dims, morton_weight=0.3, hilbert_weight=0.7, exaggeration=1.0):
    """
    Weighted bit interleaving with adaptive weights.
    """
    morton = morton_index_nd_safe(coords, bits, dims)
    hilbert = hilbert_index_nd_safe(coords, bits, dims)

    combined = (morton * morton_weight + hilbert * hilbert_weight).astype(jnp.float64)

    # Apply exaggeration to gaps
    if exaggeration != 1.0:
        # Normalize to [0, 1] before exponentiation
        min_val = jnp.min(combined)
        max_val = jnp.max(combined)
        normalized = (combined - min_val) / (max_val - min_val + 1e-9)
        normalized = normalized ** exaggeration
        combined = min_val + normalized * (max_val - min_val)

    return combined.astype(jnp.uint64)


# ============================================================
# WEIGHT OPTIMIZER (FAST GRID SEARCH)
# ============================================================

def optimize_mixing_weights(X_low, bits, dims, n_samples=1000, n_trials=20):
    """
    Fast optimization of mixing weights using sampling and grid search.
    Optimizes for maximum cluster separation.

    Args:
        X_low: Reduced data
        bits: Bits per dimension
        dims: Number of dimensions
        n_samples: Number of points to sample for optimization
        n_trials: Number of weight combinations to try

    Returns:
        best_weights: Tuple of (morton_weight, hilbert_weight, exaggeration)
        best_score: Separation score
    """
    N = len(X_low)

    # Sample points for faster optimization
    if N > n_samples:
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, N, shape=(n_samples,), replace=False)
        X_sample = X_low[idx]
    else:
        X_sample = X_low

    # Grid search over weight combinations
    best_score = -jnp.inf
    best_params = (0.3, 0.7, 1.0)

    # Generate candidate weight combinations
    candidate_weights = [
        (0.2, 0.8, 1.0),   # More Hilbert
        (0.3, 0.7, 1.0),   # Balanced (default)
        (0.4, 0.6, 1.0),   # More Morton
        (0.25, 0.75, 1.5), # Exaggerated
        (0.35, 0.65, 2.0), # More exaggerated
        (0.2, 0.8, 2.5),   # Highly exaggerated
        (0.3, 0.7, 3.0),   # Very exaggerated
        (0.15, 0.85, 1.5), # Strong Hilbert + exaggeration
        (0.4, 0.6, 2.0),   # Balanced with exaggeration
    ]

    for morton_w, hilbert_w, exag in candidate_weights[:n_trials]:
        # Compute indices
        indices = jax.vmap(
            lambda p: weighted_bit_interleaved_adaptive(p, bits, dims, morton_w, hilbert_w, exag)
        )(X_sample)

        # Compute gaps
        sorted_idx = jnp.sort(indices)
        gaps = (sorted_idx[1:] - sorted_idx[:-1]).astype(jnp.float64)

        # Get separation score
        stats = compute_gap_statistics(gaps)
        score = stats['separation_score']

        if score > best_score:
            best_score = score
            best_params = (morton_w, hilbert_w, exag)

    return best_params, best_score


# ============================================================
# ENHANCED OPTIMAL HYBRID MAP WITH STATISTICAL OPTIMIZATION
# ============================================================

class EnhancedOptimalHybridMap(OptimalHybridMap):
    """
    Enhanced optimal hybrid map with statistical optimization for better cluster separation.
    """

    def __init__(self,
                 n_components=2,
                 reduced_dims=32,
                 bits=6,
                 window=16,
                 ensemble_size=3,
                 optimize_weights=True,
                 exaggeration_auto=True,
                 random_state=42):
        super().__init__(n_components, reduced_dims, bits, window, ensemble_size, random_state)
        self.optimize_weights = optimize_weights
        self.exaggeration_auto = exaggeration_auto
        self.optimized_morton_weight_ = 0.3
        self.optimized_hilbert_weight_ = 0.7
        self.optimized_exaggeration_ = 1.0
        self.separation_score_ = None

    def fit_transform(self, X):
        """Fit and transform with statistical optimization."""
        key = jax.random.PRNGKey(self.random_state)

        # Convert to JAX array
        X_jax = jnp.array(X, dtype=jnp.float32)

        # Step 1: Dimensionality reduction
        if X.shape[1] > self.reduced_dims:
            X_low = randomized_svd_safe(X_jax, self.reduced_dims, key)
        else:
            X_low = X_jax
            self.reduced_dims = X.shape[1]

        # Step 2: Compute safe bits
        self.safe_bits_ = compute_safe_bits(self.reduced_dims, self.bits)

        # Step 3: Optimize mixing weights (if enabled)
        if self.optimize_weights:
            print("Optimizing mixing weights for better cluster separation...")
            self.optimized_morton_weight_, self.optimized_hilbert_weight_, self.optimized_exaggeration_
            best_params, best_score = optimize_mixing_weights(
                X_low, self.safe_bits_, self.reduced_dims,
                n_samples=min(2000, len(X_low)),
                n_trials=10
            )
            self.optimized_morton_weight_, self.optimized_hilbert_weight_, self.optimized_exaggeration_ = best_params
            self.separation_score_ = best_score
            print(f"  Best weights: Morton={self.optimized_morton_weight_:.2f}, Hilbert={self.optimized_hilbert_weight_:.2f}")
            print(f"  Exaggeration: {self.optimized_exaggeration_:.2f}")
            print(f"  Separation score: {best_score:.4f}")

        # Step 4: Normalize
        mins = jnp.min(X_low, axis=0)
        maxs = jnp.max(X_low, axis=0)

        # Step 5: Weighted bit-interleaved embedding with optimized parameters
        X_embedded, indices = self._weighted_bit_interleaved_embedding_optimized(
            X_low, mins, maxs, key
        )

        # Step 6: Cluster detection with adaptive threshold
        self.labels_ = detect_clusters_from_curve_adaptive(indices, exaggeration=self.optimized_exaggeration_)
        self.embedded_ = X_embedded

        return np.asarray(self.embedded_[:, :self.n_components])

    def _weighted_bit_interleaved_embedding_optimized(self, X, mins, maxs, key):
        """
        Weighted bit-interleaved embedding with optimized parameters.
        """
        N, D = X.shape
        bits = self.safe_bits_
        window = self.window

        # Safe grid computation
        scale = (2.0**bits - 1.0) / (maxs - mins + 1e-9)
        grid = jnp.floor((X - mins) * scale).astype(jnp.uint64)
        max_val = 2**bits - 1
        grid = jnp.clip(grid, 0, max_val)

        embeddings = []
        all_indices = []
        subkey = key

        for i in range(self.ensemble_size):
            subkey, rot_key, perm_key = jax.random.split(subkey, 3)

            # Random rotation and permutation
            X_rot = random_rotation_safe(rot_key, X)
            perm = jax.random.permutation(perm_key, D)
            X_perm = X_rot[:, perm]

            # Recompute grid for rotated data
            mins_rot = jnp.min(X_perm, axis=0)
            maxs_rot = jnp.max(X_perm, axis=0)
            scale_rot = (2.0**bits - 1.0) / (maxs_rot - mins_rot + 1e-9)
            grid_rot = jnp.floor((X_perm - mins_rot) * scale_rot).astype(jnp.uint64)
            grid_rot = jnp.clip(grid_rot, 0, max_val)

            # Compute weighted indices with optimized parameters
            indices = jax.vmap(
                lambda p: weighted_bit_interleaved_adaptive(
                    p, bits, D,
                    self.optimized_morton_weight_,
                    self.optimized_hilbert_weight_,
                    self.optimized_exaggeration_ if self.exaggeration_auto else 1.0
                )
            )(grid_rot)

            # Sort and smooth
            order = jnp.argsort(indices)
            X_sorted = X_perm[order]
            X_smooth = gaussian_smooth_static(X_sorted, window)

            # Restore original order
            inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
            X_restored = X_smooth[inv_order]

            embeddings.append(X_restored)

            if i == 0:
                all_indices.append(indices)

        return jnp.mean(jnp.stack(embeddings), axis=0), all_indices[0]

    def get_optimization_stats(self):
        """Return optimization statistics."""
        return {
            'morton_weight': self.optimized_morton_weight_,
            'hilbert_weight': self.optimized_hilbert_weight_,
            'exaggeration': self.optimized_exaggeration_,
            'separation_score': self.separation_score_
        }


def detect_clusters_from_curve_adaptive(indices, exaggeration=1.0, min_cluster_size=3):
    """
    Adaptive cluster detection with exaggeration.
    """
    # Sort indices
    order = jnp.argsort(indices)
    sorted_indices = indices[order]

    # Compute gaps with exaggeration
    gaps = (sorted_indices[1:] - sorted_indices[:-1]).astype(jnp.float64)
    if exaggeration != 1.0:
        gaps = gaps ** exaggeration

    # Use adaptive threshold based on gap distribution
    median_gap = jnp.median(gaps)
    q75 = jnp.percentile(gaps, 75)
    q25 = jnp.percentile(gaps, 25)
    iqr = q75 - q25

    # Dynamic threshold: median + 1.5 * IQR (similar to box plot)
    gap_threshold = median_gap + 1.5 * iqr

    # Find cluster boundaries
    boundaries = jnp.where(gaps > gap_threshold)[0] + 1

    # Build cluster labels
    labels = jnp.zeros(len(indices), dtype=jnp.int32)

    if len(boundaries) > 0:
        # First segment
        first_end = boundaries[0]
        if first_end >= min_cluster_size:
            labels = labels.at[order[:first_end]].set(0)
            start = first_end
            cluster_id = 1
        else:
            start = first_end
            cluster_id = 0

        # Remaining segments
        for i in range(1, len(boundaries)):
            end = boundaries[i]
            if end - start >= min_cluster_size:
                labels = labels.at[order[start:end]].set(cluster_id)
                cluster_id += 1
            start = end

        # Last segment
        if len(indices) - start >= min_cluster_size:
            labels = labels.at[order[start:]].set(cluster_id)

    return labels


# ============================================================
# MAIN FUNCTION
# ============================================================

def detmap_embedding_optimized(X, n_components=2, reduced_dims=32, bits=6,
                                window=16, ensemble_size=3, optimize_weights=True,
                                random_state=42):
    """
    Optimized DetMap embedding with automatic weight optimization.
    """
    mapper = OptimalHybridMapOptimized(
        n_components=n_components,
        reduced_dims=reduced_dims,
        bits=bits,
        window=window,
        ensemble_size=ensemble_size,
        optimize_weights=optimize_weights,
        random_state=random_state
    )

    embedded = mapper.fit_transform(X)
    labels = mapper.get_labels()
    stats = mapper.get_optimization_stats()

    return embedded, labels, stats



# ============================================================
# NON-LINEAR GAP ENHANCEMENT
# ============================================================

@partial(jax.jit, static_argnames=("method", "power"))
def enhance_gaps_nonlinear(gaps, method="sigmoid", power=2.0):
    """
    Apply non-linear transformation to gaps to enhance separation.

    Methods:
    - "sigmoid": S-shaped curve that amplifies medium gaps
    - "power": Power law that amplifies larger gaps more
    - "log": Log transform that compresses small gaps
    - "tanh": Hyperbolic tangent for smooth amplification
    """
    # Normalize gaps to [0, 1] range
    gaps_min = jnp.min(gaps)
    gaps_max = jnp.max(gaps)
    gaps_norm = (gaps - gaps_min) / (gaps_max - gaps_min + 1e-9)

    if method == "sigmoid":
        # Sigmoid: amplifies mid-range gaps, compresses extremes
        # Center around median
        median = jnp.median(gaps_norm)
        enhanced = 1.0 / (1.0 + jnp.exp(-10.0 * (gaps_norm - median)))

    elif method == "power":
        # Power law: amplify larger gaps exponentially
        enhanced = gaps_norm ** power

    elif method == "log":
        # Log: compress small gaps, emphasize differences
        enhanced = jnp.log1p(gaps_norm * 10) / jnp.log1p(10.0)

    elif method == "tanh":
        # Tanh: smooth amplification
        enhanced = jnp.tanh(gaps_norm * 3.0)

    elif method == "adaptive_power":
        # Adaptive: more aggressive for larger gaps
        threshold = jnp.percentile(gaps_norm, 70)
        enhanced = jnp.where(gaps_norm > threshold,
                            gaps_norm ** (power * 1.5),
                            gaps_norm ** power)
    else:
        enhanced = gaps_norm

    # Scale back to original range
    return gaps_min + enhanced * (gaps_max - gaps_min)


@partial(jax.jit, static_argnames=("n_levels",))
def multiscale_gap_analysis(gaps, n_levels=3):
    """
    Analyze gaps at multiple scales to detect hierarchical structure.
    Returns weighted combination of gaps at different scales.
    """
    enhanced_gaps = gaps.copy()

    for level in range(1, n_levels):
        # Coarse-grain at different scales
        scale = 2 ** level
        if len(gaps) > scale:
            coarse = jnp.convolve(gaps, jnp.ones(scale)/scale, mode='valid')
            # Interpolate back to original size
            coarse_expanded = jnp.repeat(coarse, scale)[:len(gaps)]
            # Weighted combination (higher weight for finer scales)
            weight = 1.0 / (level + 1)
            enhanced_gaps = enhanced_gaps * (1 - weight) + coarse_expanded * weight

    return enhanced_gaps


# ============================================================
# LOCAL DENSITY ADAPTATION
# ============================================================

@partial(jax.jit, static_argnames=("window",))
def compute_local_density(X_sorted, window=32):
    """
    Compute local density along the curve for adaptive separation.
    """
    N = X_sorted.shape[0]

    # Compute pairwise distances along curve
    diff = jnp.linalg.norm(X_sorted[1:] - X_sorted[:-1], axis=1)
    diff = jnp.concatenate([diff[:1], diff])

    # Smooth density estimate
    from jax.scipy.signal import convolve
    kernel = jnp.ones(window) / window
    density = convolve(diff, kernel, mode='same')
    density = 1.0 / (density + 1e-9)

    # Normalize
    density = density / (jnp.max(density) + 1e-9)

    return density


@partial(jax.jit, static_argnames=("strength",))
def apply_density_adaptive_gaps(gaps, density, strength=1.0):
    """
    Apply density-adaptive scaling to gaps.
    Low-density regions get amplified gaps (better separation).
    """
    # Inverse density weighting: sparse regions get larger gaps
    adaptive_factor = 1.0 + strength * (1.0 - density[:-1])
    return gaps * adaptive_factor


# ============================================================
# NON-LINEAR CURVE INDEX ENHANCEMENT
# ============================================================

@partial(jax.jit, static_argnames=("bits", "dims", "exaggeration", "nonlinear_strength"))
def nonlinear_hybrid_index(coords, bits, dims, exaggeration=2.0, nonlinear_strength=1.5):
    """
    Non-linear hybrid index with enhanced separation.
    """
    # Base hybrid index
    morton = morton_index_nd_safe(coords, bits, dims)
    hilbert = hilbert_index_nd_safe(coords, bits, dims)

    # Adaptive weights based on coordinate distribution
    # This creates non-linear mixing
    morton_float = morton.astype(jnp.float64)
    hilbert_float = hilbert.astype(jnp.float64)

    # Compute local variance for adaptive weighting
    coord_variance = jnp.var(coords.astype(jnp.float64))
    adaptive_weight = 0.5 + 0.3 * jnp.tanh(coord_variance - 0.5)

    combined = morton_float * (1 - adaptive_weight) + hilbert_float * adaptive_weight

    # Apply non-linear transformation
    # This enhances the differences between clusters
    combined_norm = (combined - jnp.min(combined)) / (jnp.max(combined) - jnp.min(combined) + 1e-9)

    # Non-linear stretching using error function (smooth step)
    combined_norm = 0.5 * (1.0 + jnp.tanh(nonlinear_strength * (combined_norm - 0.5)))

    # Power exaggeration
    combined_norm = combined_norm ** exaggeration

    # Scale back
    result = jnp.min(combined) + combined_norm * (jnp.max(combined) - jnp.min(combined))

    return result.astype(jnp.uint64)


# ============================================================
# ENHANCED CLUSTER DETECTION WITH NON-LINEAR GAPS
# ============================================================

def detect_clusters_nonlinear(indices, method="adaptive_power",
                               power=2.0, density_strength=1.0,
                               min_cluster_size=5):
    """
    Enhanced cluster detection using non-linear gap analysis.
    """
    # Sort indices
    order = jnp.argsort(indices)
    sorted_indices = indices[order]

    # Compute base gaps
    gaps = (sorted_indices[1:] - sorted_indices[:-1]).astype(jnp.float64)

    # Apply non-linear enhancement
    gaps_enhanced = enhance_gaps_nonlinear(gaps, method=method, power=power)

    # Apply multiscale analysis
    gaps_multiscale = multiscale_gap_analysis(gaps_enhanced, n_levels=3)

    # Adaptive threshold based on enhanced gaps
    median_gap = jnp.median(gaps_multiscale)
    q75 = jnp.percentile(gaps_multiscale, 75)
    q25 = jnp.percentile(gaps_multiscale, 25)
    iqr = q75 - q25

    # Dynamic threshold (more aggressive for non-linear)
    gap_threshold = median_gap + 1.2 * iqr

    # Find cluster boundaries
    boundaries = jnp.where(gaps_multiscale > gap_threshold)[0] + 1

    # Build cluster labels
    labels = jnp.zeros(len(indices), dtype=jnp.int32)

    if len(boundaries) > 0:
        # First segment
        first_end = boundaries[0]
        if first_end >= min_cluster_size:
            labels = labels.at[order[:first_end]].set(0)
            start = first_end
            cluster_id = 1
        else:
            start = first_end
            cluster_id = 0

        # Remaining segments
        for i in range(1, len(boundaries)):
            end = boundaries[i]
            if end - start >= min_cluster_size:
                labels = labels.at[order[start:end]].set(cluster_id)
                cluster_id += 1
            start = end

        # Last segment
        if len(indices) - start >= min_cluster_size:
            labels = labels.at[order[start:]].set(cluster_id)

    return labels, gaps_multiscale


# ============================================================
# HYBRID MAP WITH NON-LINEAR SEPARATION
# ============================================================

class NonlinearHybridMap(OptimalHybridMap):
    """
    hybrid map with non-linear separation enhancement.
    Best for complex datasets like MNIST.
    """

    def __init__(self,
                 n_components=2,
                 reduced_dims=32,
                 bits=6,
                 window=16,
                 ensemble_size=3,
                 gap_method="adaptive_power",
                 nonlinear_strength=1.5,
                 gap_power=2.0,
                 density_strength=0.5,
                 optimize_weights=True,
                 random_state=42):
        super().__init__(n_components, reduced_dims, bits, window, ensemble_size, random_state)
        self.gap_method = gap_method
        self.nonlinear_strength = nonlinear_strength
        self.gap_power = gap_power
        self.density_strength = density_strength
        self.optimize_weights = optimize_weights



    def transform(self, X):
        """Fit and transform with non-linear separation."""
        key = jax.random.PRNGKey(self.random_state)

        # Convert to JAX array
        X_jax = jnp.array(X, dtype=jnp.float32)

        # Step 1: Dimensionality reduction
        if X.shape[1] > self.reduced_dims:
            X_low = randomized_svd_safe(X_jax, self.reduced_dims, key)
        else:
            X_low = X_jax
            self.reduced_dims = X.shape[1]

        # Step 2: Compute safe bits
        self.safe_bits_ = compute_safe_bits(self.reduced_dims, self.bits)

        # Step 3: Optimize weights (optional)
        if self.optimize_weights:
            self._optimize_nonlinear_weights(X_low)

        # Step 4: Normalize
        mins = jnp.min(X_low, axis=0)
        maxs = jnp.max(X_low, axis=0)

        # Step 5: Non-linear embedding
        X_embedded, indices, gaps = self._nonlinear_embedding(
            X_low, mins, maxs, key
        )

        # Step 6: Enhanced cluster detection
        self.labels_, _ = detect_clusters_nonlinear(
            indices,
            method=self.gap_method,
            power=self.gap_power,
            density_strength=self.density_strength,
            min_cluster_size=3
        )

        self.embedded_ = X_embedded
        self.gaps_ = gaps

        return np.asarray(self.embedded_[:, :self.n_components])

    def _optimize_nonlinear_weights(self, X_low):
        """Fast optimization for non-linear parameters."""
        N = len(X_low)
        n_samples = min(2000, N)

        # Sample points
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, N, shape=(n_samples,), replace=False)
        X_sample = X_low[idx]

        bits = self.safe_bits_
        dims = self.reduced_dims

        # Test different configurations
        configs = [
            (0.3, 0.7, 1.0, "power"),
            (0.3, 0.7, 1.5, "adaptive_power"),
            (0.4, 0.6, 2.0, "adaptive_even_power"),
            (0.2, 0.8, 1.5, "sigmoid"),
            (0.25, 0.75, 2.0, "tanh"),
        ]

        best_score = -jnp.inf
        best_config = (0.3, 0.7, 1.0, "adaptive_power")

        for m_w, h_w, exag, method in configs:
            # Compute indices
            indices = jax.vmap(
                lambda p: nonlinear_hybrid_index(p, bits, dims, exag, 1.5)
            )(X_sample)

            # Sort and compute gaps
            sorted_idx = jnp.sort(indices)
            gaps = (sorted_idx[1:] - sorted_idx[:-1]).astype(jnp.float64)
            gaps_enhanced = enhance_gaps_nonlinear(gaps, method=method, power=2.0)

            # Score based on gap distribution
            variance = jnp.var(gaps_enhanced)
            kurtosis = jnp.mean((gaps_enhanced - jnp.mean(gaps_enhanced))**4) / (jnp.var(gaps_enhanced)**2 + 1e-9)
            score = variance * 0.5 + (kurtosis - 3) * 0.5

            if score > best_score:
                best_score = score
                best_config = (m_w, h_w, exag, method)
                self.gap_method = method

        self.optimized_morton_weight_, self.optimized_hilbert_weight_, self.optimized_exaggeration_, _ = best_config

        print(f"Optimized: method={self.gap_method}, exaggeration={self.optimized_exaggeration_:.2f}")

    def _nonlinear_embedding(self, X, mins, maxs, key):
        """Non-linear embedding with enhanced separation."""
        N, D = X.shape
        bits = self.safe_bits_
        window = self.window

        # Safe grid computation
        scale = (2.0**bits - 1.0) / (maxs - mins + 1e-9)
        grid = jnp.floor((X - mins) * scale).astype(jnp.uint64)
        max_val = 2**bits - 1
        grid = jnp.clip(grid, 0, max_val)

        embeddings = []
        all_indices = []
        all_gaps = []
        subkey = key

        for i in range(self.ensemble_size):
            subkey, rot_key, perm_key = jax.random.split(subkey, 3)

            # Random rotation and permutation
            X_rot = random_rotation_safe(rot_key, X)
            perm = jax.random.permutation(perm_key, D)
            X_perm = X_rot[:, perm]

            # Recompute grid
            mins_rot = jnp.min(X_perm, axis=0)
            maxs_rot = jnp.max(X_perm, axis=0)
            scale_rot = (2.0**bits - 1.0) / (maxs_rot - mins_rot + 1e-9)
            grid_rot = jnp.floor((X_perm - mins_rot) * scale_rot).astype(jnp.uint64)
            grid_rot = jnp.clip(grid_rot, 0, max_val)

            # Non-linear hybrid indices
            indices = jax.vmap(
                lambda p: nonlinear_hybrid_index(
                    p, bits, D,
                    self.optimized_exaggeration_ if hasattr(self, 'optimized_exaggeration_') else 1.5,
                    self.nonlinear_strength
                )
            )(grid_rot)

            # Sort and smooth
            order = jnp.argsort(indices)
            X_sorted = X_perm[order]
            X_smooth = gaussian_smooth_static(X_sorted, window)

            # Restore order
            inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
            X_restored = X_smooth[inv_order]

            embeddings.append(X_restored)

            if i == 0:
                all_indices.append(indices)
                # Compute enhanced gaps for reporting
                sorted_idx = jnp.sort(indices)
                gaps = (sorted_idx[1:] - sorted_idx[:-1]).astype(jnp.float64)
                all_gaps.append(enhance_gaps_nonlinear(gaps, method=self.gap_method, power=self.gap_power))

        return jnp.mean(jnp.stack(embeddings), axis=0), all_indices[0], all_gaps[0]


# ============================================================
# MAIN FUNCTION
# ============================================================

def detmap_nonlinear(X, n_components=2, reduced_dims=32, bits=5,
                     gap_method="adaptive_power", nonlinear_strength=1.5,
                     random_state=42):
    """
    Non-linear DetMap embedding for complex datasets like MNIST.

    Args:
        X: Input data (samples, features)
        n_components: Output dimensions
        reduced_dims: Target reduced dimensions
        bits: Bits per dimension (use 4-5 for MNIST)
        gap_method: Non-linear gap enhancement method
        nonlinear_strength: Strength of non-linear transformation (1.0-2.0)
        random_state: Random seed

    Returns:
        embedded: Coordinates
        labels: Cluster labels
    """
    mapper = NonlinearHybridMap(
        n_components=n_components,
        reduced_dims=reduced_dims,
        bits=bits,
        window=12,
        ensemble_size=2,  # Reduced for speed
        gap_method=gap_method,
        nonlinear_strength=nonlinear_strength,
        gap_power=2.0,
        density_strength=0.5,
        optimize_weights=True,
        random_state=random_state
    )

    embedded = mapper.fit_transform(X)
    labels = mapper.get_labels()

    return embedded, labels


# ============================================================
# TEST ON MNIST-LIKE DATA
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # Generate MNIST-like data (10 clusters with varying difficulty)
    np.random.seed(42)
    n_points = 5000
    n_digits = 5  # 5 classes for testing

    X_list = []
    centers = []

    # Create clusters with varying separation
    for i in range(n_digits):
        # Different means for different digits
        mean = np.random.randn(50) * (i + 1) * 0.8
        # Varying variance to simulate different digit complexities
        std = 0.5 + i * 0.2
        cluster = np.random.randn(n_points // n_digits, 50) * std + mean
        X_list.append(cluster)
        centers.append(mean[:2])

    X = np.vstack(X_list)
    y_true = np.repeat(np.arange(n_digits), n_points // n_digits)

    print(f"Data shape: {X.shape}")

    # Test different configurations
    methods = ["power", "adaptive_power", "sigmoid", "tanh"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]

        start = time.time()
        X_embedded, labels = detmap_nonlinear(
            X, n_components=2, reduced_dims=16, bits=4,
            gap_method=method, nonlinear_strength=1.5,
            random_state=42
        )
        elapsed = time.time() - start

        # Plot
        scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                            c=labels, cmap='tab10', alpha=0.6, s=8)
        ax.set_title(f"Method: {method}\nTime: {elapsed:.2f}s, Clusters: {len(np.unique(labels))}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # EXAMPLE USAGE
    # ============================================================
    import matplotlib.pyplot as plt
    import time

    # Generate challenging data with overlapping clusters
    np.random.seed(42)
    n_points = 3000
    n_clusters = 4

    X_list = []
    centers = [8, 3, -2, -7]
    for i, center in enumerate(centers):
        cluster = np.random.randn(n_points // n_clusters, 15) * 0.8 + center
        X_list.append(cluster)

    X = np.vstack(X_list)
    print(f"Data shape: {X.shape}")

    # Test with and without optimization
    print("\n=== Without Optimization ===")
    start = time.time()
    mapper1 = OptimalHybridMap(n_components=2, reduced_dims=8, bits=3, window=16, ensemble_size=3)
    X1 = mapper1.fit_transform(X)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Clusters: {len(np.unique(mapper1.get_labels()))}")

    print("\n=== With Statistical Optimization ===")
    start = time.time()
    mapper2 = OptimalHybridMapOptimized(n_components=2, reduced_dims=8, bits=3, window=16,
                                         ensemble_size=3, optimize_weights=True)
    X2 = mapper2.fit_transform(X)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Clusters: {len(np.unique(mapper2.get_labels()))}")
    stats = mapper2.get_optimization_stats()
    print(f"Optimized: Morton={stats['morton_weight']:.2f}, Hilbert={stats['hilbert_weight']:.2f}, Exag={stats['exaggeration']:.2f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X1[:, 0], X1[:, 1], c=mapper1.get_labels(), cmap='tab10', alpha=0.7, s=15)
    axes[0].set_title("Without Optimization")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].scatter(X2[:, 0], X2[:, 1], c=mapper2.get_labels(), cmap='tab10', alpha=0.7, s=15)
    axes[1].set_title("With Statistical Optimization")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()

    # ============================================================
    # EXAMPLE USAGE
    # ============================================================

    import matplotlib.pyplot as plt

    # Generate synthetic data
    np.random.seed(42)
    n_points = 2000
    n_clusters = 5

    X_list = []
    for i in range(n_clusters):
        center = np.random.randn(10) * 5
        cluster = center + np.random.randn(n_points // n_clusters, 10) * 0.5
        X_list.append(cluster)

    X = np.vstack(X_list)
    print(f"Data shape: {X.shape}")

    # Create optimal map
    detmap = OptimalHybridMap(
        n_components=2,
        reduced_dims=2,
        bits=3,
        window=16,
        ensemble_size=3,
        random_state=42
    )

    print(f"Safe bits for {detmap.reduced_dims} dimensions: {detmap.safe_bits_}")

    # Transform data
    X_embedded = detmap.fit_transform(X)
    labels = detmap.get_labels()

    print(f"Embedding shape: {X_embedded.shape}")
    print(f"Number of clusters detected: {len(np.unique(labels))}")

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                         c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter)
    plt.title("Optimal Hybrid Map")
    plt.show()
