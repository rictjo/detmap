"""
jax_cluster_separation.py - Pure JAX implementation of cluster separation
for space-filling curve embeddings
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import numpy as np

from ..base import BaseMap
import numpy as np

# ============================================================
# JAX-ONLY K-MEANS CLUSTERING
# ============================================================

@partial(jax.jit, static_argnames=("n_clusters", "max_iters", "batch_size"))
def jax_kmeans(X, n_clusters, max_iters=100, batch_size=1000, key=jax.random.PRNGKey(0)):
    """
    Pure JAX implementation of k-means clustering.
    Uses mini-batch for large datasets.
    """
    N, D = X.shape

    # Initialize centroids using k-means++ (JAX version)
    key, subkey = jax.random.split(key)
    centroids = jax_kmeans_plus_plus(X, n_clusters, subkey)

    def kmeans_step(carry, batch):
        centroids, = carry
        batch = batch

        # Assign points to nearest centroid
        distances = jnp.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = jnp.argmin(distances, axis=1)

        # Update centroids
        new_centroids = jnp.zeros_like(centroids)
        counts = jnp.zeros(n_clusters, dtype=jnp.int32)

        def update_cluster(i, state):
            new_centroids, counts = state
            mask = labels == i
            if jnp.any(mask):
                new_centroids = new_centroids.at[i].set(jnp.mean(batch[mask], axis=0))
                counts = counts.at[i].set(jnp.sum(mask))
            return (new_centroids, counts)

        new_centroids, counts = lax.fori_loop(0, n_clusters, update_cluster, (new_centroids, counts))

        return (new_centroids,), (labels, counts)

    # Mini-batch processing
    n_batches = (N + batch_size - 1) // batch_size

    def process_epoch(carry, _):
        centroids, = carry

        # Shuffle data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        X_shuffled = X[perm]

        # Process batches
        def process_batch(carry, batch_start):
            centroids, = carry
            batch = X_shuffled[batch_start:batch_start + batch_size]
            (new_centroids,), (_, counts) = kmeans_step((centroids,), batch)
            # Exponential moving average update
            alpha = jnp.minimum(1.0, batch_size / (batch_size + 100))
            centroids = (1 - alpha) * centroids + alpha * new_centroids
            return (centroids,), None

        batch_starts = jnp.arange(0, N, batch_size)
        (centroids,), _ = lax.scan(process_batch, (centroids,), batch_starts)
        return (centroids,), None

    (centroids,), _ = lax.scan(process_epoch, (centroids,), None, length=max_iters)

    # Final assignment
    distances = jnp.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = jnp.argmin(distances, axis=1)

    return labels, centroids


def jax_kmeans_plus_plus(X, n_clusters, key):
    """K-means++ initialization in pure JAX."""
    N, D = X.shape

    # Choose first centroid randomly
    key, subkey = jax.random.split(key)
    idx = jax.random.randint(subkey, (1,), 0, N)
    centroids = X[idx]

    def select_next(i, centroids):
        # Compute distances to nearest centroid
        distances = jnp.min(jnp.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
        distances = distances + 1e-12

        # Choose next centroid with probability proportional to distance^2
        probs = distances / jnp.sum(distances)
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, N, shape=(1,), p=probs)
        return jnp.concatenate([centroids, X[idx]], axis=0)

    centroids = lax.fori_loop(1, n_clusters, select_next, centroids)
    return centroids


# ============================================================
# ADAPTIVE LOG TRANSFORM
# ============================================================

@partial(jax.jit, static_argnames=("k_neighbors",))
def adaptive_log_transform(X, k_neighbors=10, epsilon=1e-8):
    """
    Pure JAX implementation of adaptive log transform.
    Computes local scale using approximate nearest neighbors.
    """
    N, D = X.shape

    # For large N, use random sampling for efficiency
    if N > 10000:
        sample_size = min(5000, N)
        key = jax.random.PRNGKey(0)
        sample_idx = jax.random.choice(key, N, shape=(sample_size,), replace=False)
        X_sample = X[sample_idx]

        # Compute local scale for sample
        distances = compute_pairwise_distances(X_sample)
        # Use k-th neighbor distance as local scale
        k_small = min(k_neighbors, sample_size - 1)
        kth_distances = jnp.sort(distances, axis=1)[:, k_small]

        # Interpolate for all points
        from jax.scipy.spatial.distance import cdist
        dist_to_sample = cdist(X, X_sample)
        nearest_sample_idx = jnp.argmin(dist_to_sample, axis=1)
        local_scale = kth_distances[nearest_sample_idx]
    else:
        distances = compute_pairwise_distances(X)
        k_small = min(k_neighbors, N - 1)
        local_scale = jnp.sort(distances, axis=1)[:, k_small]

    local_scale = jnp.maximum(local_scale[:, None], epsilon)

    # Apply adaptive log transform
    X_scaled = X / local_scale
    X_transformed = jnp.sign(X_scaled) * jnp.log1p(jnp.abs(X_scaled))

    return X_transformed


@partial(jax.jit, static_argnames=("max_samples",))
def compute_pairwise_distances(X, max_samples=10000):
    """Efficient pairwise distance computation for large matrices."""
    N, D = X.shape

    if N > max_samples:
        # Use random sampling for large datasets
        key = jax.random.PRNGKey(0)
        idx = jax.random.choice(key, N, shape=(max_samples,), replace=False)
        X_subset = X[idx]
        return jnp.sum((X_subset[:, None, :] - X_subset[None, :, :]) ** 2, axis=2)

    # Full computation for smaller datasets
    return jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)


# ============================================================
# MULTISCALE SEPARATION TRANSFORM
# ============================================================

@partial(jax.jit, static_argnames=("scales", "exaggeration_factor"))
def multiscale_separation_transform(X, scales=jnp.array([0.1, 0.5, 1.0, 2.0, 5.0]),
                                    exaggeration_factor=2.0):
    """
    Pure JAX implementation of multiscale separation transform.
    """
    N, D = X.shape
    X_norm = X / (jnp.std(X, axis=0, keepdims=True) + 1e-8)

    # Compute distance to k-nearest neighbors at different scales
    features = [X_norm]  # Original coordinates as baseline

    for scale in scales:
        k = jnp.clip(jnp.array(scale * N / 100, dtype=jnp.int32), 3, N - 1)

        # Compute pairwise distances (sampled for efficiency)
        if N > 5000:
            # Use random sampling for large datasets
            sample_size = min(2000, N)
            key = jax.random.PRNGKey(0)
            sample_idx = jax.random.choice(key, N, shape=(sample_size,), replace=False)
            X_sample = X_norm[sample_idx]
            dist_to_sample = compute_pairwise_distances(X_norm, X_sample)

            # Get k nearest neighbors
            k_small = jnp.minimum(k, sample_size)
            kth_distances = jnp.sort(dist_to_sample, axis=1)[:, k_small - 1]
        else:
            distances = compute_pairwise_distances(X_norm)
            kth_distances = jnp.sort(distances, axis=1)[:, k - 1]

        # Density feature (points with many close neighbors get compressed)
        avg_dist = jnp.mean(kth_distances, keepdims=True)
        density_feature = jnp.exp(-kth_distances[:, None] / (scale * avg_dist + 1e-8))

        # Separation feature (points in sparse regions get pushed outward)
        separation_feature = (1 - density_feature) ** exaggeration_factor

        features.append(density_feature)
        features.append(separation_feature)

    return jnp.concatenate(features, axis=1)


#@partial(jax.jit, static_argnames=("n_neighbors",))
@jax.jit
def compute_pairwise_distances(X, Y=None):
    """Compute pairwise distances between X and Y."""
    if Y is None:
        Y = X
    return jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)


# ============================================================
# CLUSTER EXAGGERATION TRANSFORM
# ============================================================

@partial(jax.jit, static_argnames=("n_clusters", "exaggeration_factor"))
def cluster_exaggeration_transform(X, n_clusters=20, exaggeration_factor=2.0,
                                   key=jax.random.PRNGKey(0)):
    """
    Exaggerate distances from cluster centers.
    """
    N, D = X.shape

    # Adjust number of clusters based on dataset size
    n_clusters = jnp.minimum(n_clusters, jnp.maximum(5, N // 200))

    # Cluster the data
    labels, centroids = jax_kmeans(X, n_clusters, key=key)

    # Exaggerate distances
    def exaggerate_point(i):
        point = X[i]
        label = labels[i]
        center = centroids[label]
        displacement = point - center
        return center + displacement * exaggeration_factor

    X_exaggerated = jax.vmap(exaggerate_point)(jnp.arange(N))

    return X_exaggerated, labels, centroids


# ============================================================
# SPACE-FILLING CURVE UTILITIES
# ============================================================

@partial(jax.jit, static_argnames=("bits",))
def morton_index_nd(coords, bits):
    """Morton (Z-order) curve index."""
    dims = coords.shape[0]
    coords = coords.astype(jnp.uint64)

    max_bits_per_dim = 63 // dims
    if bits > max_bits_per_dim:
        bits = max_bits_per_dim

    def spread_bits(x):
        x = x & ((1 << bits) - 1)
        x = (x | (x << 32)) & 0x00000000ffffffff
        x = (x | (x << 16)) & 0x0000ffff0000ffff
        x = (x | (x << 8)) & 0x00ff00ff00ff00ff
        x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f
        x = (x | (x << 2)) & 0x3333333333333333
        x = (x | (x << 1)) & 0x5555555555555555
        return x

    result = jnp.uint64(0)
    for d in range(dims):
        result |= spread_bits(coords[d]) << d
    return result


@partial(jax.jit, static_argnames=("bits",))
def hilbert_index_nd(coords, bits):
    """Hilbert curve index."""
    dims = coords.shape[0]
    coords = coords.astype(jnp.uint64)

    max_bits_per_dim = 63 // dims
    if bits > max_bits_per_dim:
        bits = max_bits_per_dim

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

        return lax.fori_loop(1, dims, inner, x)

    x = lax.fori_loop(0, bits, gray_step, x)

    def build_index(i, state):
        x, h = state
        b = bits - 1 - i
        digit = jnp.uint64(0)
        for d in range(dims):
            digit |= ((x[d] >> b) & 1) << d
        h = (h << dims) | digit
        return (x, h)

    _, h = lax.fori_loop(0, bits, build_index, (x, jnp.uint64(0)))
    return h


#@partial(jax.jit, static_argnames=("bits",))
@jax.jit
def hybrid_interleave(morton, hilbert):
    """Interleave Morton and Hilbert indices."""
    hilbert_high = (hilbert >> 32) & 0xFFFFFFFF
    morton_low = morton & 0xFFFFFFFF
    return (hilbert_high << 32) | morton_low

def hybrid_curve_embedding(
    X,
    reduced_dims=32,
    bits=6,
    window=32,
    ensemble_size=4,
    mixing_strategy="adaptive",  # "adaptive", "interleaved", "weighted", "ensemble"
    key=jax.random.PRNGKey(0)
):
    """
    Enhanced hybrid curve embedding with multiple mixing strategies.

    mixing_strategy options:
    - "adaptive": Per-point adaptive weighting based on cluster distance
    - "interleaved": Bit-level interleaving of Morton and Hilbert
    - "weighted": Global fixed weights
    - "ensemble": Run both curves separately and combine results
    """
    N, D = X.shape

    # Preprocessing
    X_jax = jnp.array(X, dtype=jnp.float32)
    X_low = randomized_svd_reduce(X_jax, reduced_dims, key)

    mins = jnp.min(X_low, axis=0)
    maxs = jnp.max(X_low, axis=0)

    # Compute clusters for adaptive mixing
    if mixing_strategy == "adaptive":
        _, centroids = jax_kmeans(X_low, min(20, N // 100), key=key)

    embeddings = []
    subkey = key

    for i in range(ensemble_size):
        subkey, rot_key, perm_key = jax.random.split(subkey, 3)

        X_rot = random_rotation(rot_key, X_low)
        perm = jax.random.permutation(perm_key, X_rot.shape[1])
        X_perm = X_rot[:, perm]

        grid = jnp.floor(
            (X_perm - mins) / (maxs - mins + 1e-9) * (2**bits - 1)
        ).astype(jnp.uint64)

        # Compute both indices
        morton_idx = jax.vmap(lambda p: morton_index_nd(p, bits))(grid)
        hilbert_idx = jax.vmap(lambda p: hilbert_index_nd(p, bits))(grid)

        # Choose mixing strategy
        if mixing_strategy == "adaptive":
            # Adaptive per-point weighting based on cluster proximity
            cluster_dist = jnp.min(jnp.sum((X_perm[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
            weights = jnp.exp(-cluster_dist / (jnp.median(cluster_dist) + 1e-8))
            weights = jnp.clip(weights, 0.3, 0.8)
            combined_idx = (morton_idx * (1 - weights) +
                           hilbert_idx * weights).astype(jnp.uint64)

        elif mixing_strategy == "interleaved":
            # Bit-level interleaving
            combined_idx = hybrid_interleave(morton_idx, hilbert_idx)

        elif mixing_strategy == "weighted":
            # Global fixed weights (more Hilbert weight)
            combined_idx = (morton_idx * 0.3 + hilbert_idx * 0.7).astype(jnp.uint64)

        elif mixing_strategy == "ensemble":
            # Separate curves, combine at embedding level
            for curve_type, weight in [("morton", 0.4), ("hilbert", 0.6)]:
                if curve_type == "morton":
                    order = jnp.argsort(morton_idx)
                else:
                    order = jnp.argsort(hilbert_idx)

                X_sorted = X_low[order]
                X_smooth = smooth_along_curve(X_sorted, window)
                inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
                X_restored = X_smooth[inv_order]
                embeddings.append(X_restored * weight)
            continue  # Skip the regular path

        else:  # "alternate" - alternate bits from each curve
            # Take even bits from Morton, odd from Hilbert
            combined_idx = ((morton_idx & 0xAAAAAAAAAAAAAAAA) |
                           (hilbert_idx & 0x5555555555555555))

        if mixing_strategy != "ensemble":
            order = jnp.argsort(combined_idx)
            X_sorted = X_low[order]
            X_smooth = smooth_along_curve(X_sorted, window)
            inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
            embeddings.append(X_smooth[inv_order])

    # Combine ensemble
    result = jnp.mean(jnp.stack(embeddings), axis=0)

    return result

# ============================================================
# COMPLETE EMBEDDING WITH CLUSTER SEPARATION
# ============================================================
def cluster_separated_embedding(
    X,
    reduced_dims=32,
    bits=6,
    window=32,
    ensemble_size=4,
    cluster_exaggeration=2.0,
    use_adaptive_log=True,
    use_multiscale=True,
    n_clusters=None,
    mixing_strategy="adaptive",  # "adaptive", "interleaved", "weighted", "ensemble", "alternate"
    global_morton_weight=0.3,    # For "weighted" strategy (morton weight)
    key=jax.random.PRNGKey(0)
):
    """
    Complete JAX implementation of cluster-separated embedding with hybrid curve mixing.

    Args:
        X: Input data (N, D)
        reduced_dims: Target reduced dimensions
        bits: Bits per dimension for space-filling curve
        window: Smoothing window size
        ensemble_size: Number of random rotations
        cluster_exaggeration: How much to push clusters apart
        use_adaptive_log: Apply log transform to outliers
        use_multiscale: Apply multiscale separation
        n_clusters: Number of clusters (auto if None)
        mixing_strategy: How to combine Morton and Hilbert curves
            - "adaptive": Per-point weighting based on cluster distance
            - "interleaved": Bit-level interleaving (Hilbert high bits, Morton low)
            - "weighted": Global fixed weights
            - "ensemble": Run both curves separately and combine embeddings
            - "alternate": Alternate bits (even from Morton, odd from Hilbert)
        global_morton_weight: Weight for Morton in "weighted" strategy
        key: Random key

    Returns:
        embedded: Embedded coordinates (N, reduced_dims)
        labels: Cluster labels
    """
    N, D = X.shape

    # Convert to JAX array
    X_jax = jnp.array(X, dtype=jnp.float32)

    # Step 1: Adaptive log transform for outliers
    if use_adaptive_log:
        X_jax = adaptive_log_transform(X_jax)

    # Step 2: Cluster exaggeration
    if n_clusters is None:
        n_clusters = jnp.minimum(50, jnp.maximum(5, N // 200))

    X_exaggerated, labels, centroids = cluster_exaggeration_transform(
        X_jax, n_clusters, cluster_exaggeration, key
    )

    # Step 3: Multiscale separation
    if use_multiscale:
        X_processed = multiscale_separation_transform(X_exaggerated)
    else:
        X_processed = X_exaggerated

    # Step 4: Dimensionality reduction
    X_low = randomized_svd_reduce(X_processed, reduced_dims, key)

    # Step 5: Space-filling curve ensemble with hybrid mixing
    mins = jnp.min(X_low, axis=0)
    maxs = jnp.max(X_low, axis=0)

    embeddings = []
    subkey = key

    for i in range(ensemble_size):
        subkey, rot_key, perm_key = jax.random.split(subkey, 3)

        # Random rotation
        X_rot = random_rotation(rot_key, X_low)

        # Random permutation
        perm = jax.random.permutation(perm_key, X_rot.shape[1])
        X_perm = X_rot[:, perm]

        # Build grid for curve indices
        grid = jnp.floor(
            (X_perm - mins) / (maxs - mins + 1e-9) * (2**bits - 1)
        ).astype(jnp.uint64)

        # Compute both curve indices
        morton_idx = jax.vmap(lambda p: morton_index_nd(p, bits))(grid)
        hilbert_idx = jax.vmap(lambda p: hilbert_index_nd(p, bits))(grid)

        # Apply mixing strategy
        if mixing_strategy == "adaptive":
            # Adaptive per-point weighting based on cluster proximity
            # Points near cluster boundaries get more Hilbert weighting
            cluster_center_dist = jnp.min(jnp.sum((X_perm[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
            cluster_weights = jnp.exp(-cluster_center_dist / (jnp.median(cluster_center_dist) + 1e-8))
            cluster_weights = jnp.clip(cluster_weights, 0.3, 0.8)

            combined_idx = (morton_idx * (1 - cluster_weights) +
                           hilbert_idx * cluster_weights).astype(jnp.uint64)
            order = jnp.argsort(combined_idx)

        elif mixing_strategy == "interleaved":
            # Bit-level interleaving (Hilbert high bits, Morton low bits)
            combined_idx = hybrid_interleave(morton_idx, hilbert_idx)
            order = jnp.argsort(combined_idx)

        elif mixing_strategy == "weighted":
            # Global fixed weights
            global_hilbert_weight = 1.0 - global_morton_weight
            combined_idx = (morton_idx * global_morton_weight +
                           hilbert_idx * global_hilbert_weight).astype(jnp.uint64)
            order = jnp.argsort(combined_idx)

        elif mixing_strategy == "alternate":
            # Alternate bits: even from Morton, odd from Hilbert
            combined_idx = hybrid_alternate_bits(morton_idx, hilbert_idx)
            order = jnp.argsort(combined_idx)

        elif mixing_strategy == "ensemble":
            # Run both curves separately and combine at embedding level
            # Process Morton order
            morton_order = jnp.argsort(morton_idx)
            X_morton_sorted = X_low[morton_order]
            X_morton_smooth = smooth_along_curve(X_morton_sorted, window)
            inv_morton = jnp.zeros(N, dtype=jnp.int32).at[morton_order].set(jnp.arange(N))
            X_morton = X_morton_smooth[inv_morton]

            # Process Hilbert order
            hilbert_order = jnp.argsort(hilbert_idx)
            X_hilbert_sorted = X_low[hilbert_order]
            X_hilbert_smooth = smooth_along_curve(X_hilbert_sorted, window)
            inv_hilbert = jnp.zeros(N, dtype=jnp.int32).at[hilbert_order].set(jnp.arange(N))
            X_hilbert = X_hilbert_smooth[inv_hilbert]

            # Combine with adaptive weights
            cluster_center_dist = jnp.min(jnp.sum((X_perm[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
            cluster_weights = jnp.exp(-cluster_center_dist / (jnp.median(cluster_center_dist) + 1e-8))
            cluster_weights = jnp.clip(cluster_weights, 0.3, 0.8)[:, None]

            X_combined = X_morton * (1 - cluster_weights) + X_hilbert * cluster_weights
            embeddings.append(X_combined)
            continue  # Skip the regular path

        else:
            # Default: equal weighting
            combined_idx = ((morton_idx + hilbert_idx) // 2).astype(jnp.uint64)
            order = jnp.argsort(combined_idx)

        # For non-ensemble strategies, process the combined ordering
        if mixing_strategy != "ensemble":
            X_sorted = X_low[order]
            X_smooth = smooth_along_curve(X_sorted, window)
            inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
            X_restored = X_smooth[inv_order]
            embeddings.append(X_restored)

    # Average ensemble
    embedded = jnp.mean(jnp.stack(embeddings), axis=0)

    return embedded, labels


def cluster_separated_embedding_vanilla(
    X,
    reduced_dims=32,
    bits=6,
    window=32,
    ensemble_size=4,
    cluster_exaggeration=2.0,
    use_adaptive_log=True,
    use_multiscale=True,
    n_clusters=None,
    key=jax.random.PRNGKey(0)
):
    """
    Complete JAX-only implementation of cluster-separated embedding.

    Args:
        X: Input data (N, D)
        reduced_dims: Target reduced dimensions
        bits: Bits per dimension for space-filling curve
        window: Smoothing window size
        ensemble_size: Number of random rotations
        cluster_exaggeration: How much to push clusters apart
        use_adaptive_log: Apply log transform to outliers
        use_multiscale: Apply multiscale separation
        n_clusters: Number of clusters (auto if None)
        key: Random key

    Returns:
        embedded: Embedded coordinates (N, reduced_dims)
        labels: Cluster labels
    """
    N, D = X.shape

    # Convert to JAX array
    X_jax = jnp.array(X, dtype=jnp.float32)

    # Step 1: Adaptive log transform for outliers
    if use_adaptive_log:
        X_jax = adaptive_log_transform(X_jax)

    # Step 2: Cluster exaggeration
    if n_clusters is None:
        n_clusters = jnp.minimum(50, jnp.maximum(5, N // 200))

    X_exaggerated, labels, centroids = cluster_exaggeration_transform(
        X_jax, n_clusters, cluster_exaggeration, key
    )

    # Step 3: Multiscale separation
    if use_multiscale:
        X_processed = multiscale_separation_transform(X_exaggerated)
    else:
        X_processed = X_exaggerated

    # Step 4: Dimensionality reduction (using SVD)
    X_low = randomized_svd_reduce(X_processed, reduced_dims, key)

    # Step 5: Space-filling curve ensemble
    mins = jnp.min(X_low, axis=0)
    maxs = jnp.max(X_low, axis=0)

    embeddings = []
    subkey = key

    for i in range(ensemble_size):
        subkey, rot_key, perm_key = jax.random.split(subkey, 3)

        # Random rotation
        X_rot = random_rotation(rot_key, X_low)

        # Random permutation
        perm = jax.random.permutation(perm_key, X_rot.shape[1])
        X_perm = X_rot[:, perm]

        # Build hybrid curve order
        grid = jnp.floor(
            (X_perm - mins) / (maxs - mins + 1e-9) * (2**bits - 1)
        ).astype(jnp.uint64)

        # Compute both curve indices
        morton_idx = jax.vmap(lambda p: morton_index_nd(p, bits))(grid)
        hilbert_idx = jax.vmap(lambda p: hilbert_index_nd(p, bits))(grid)

        # Hybrid index with cluster-aware weighting
        # Points near cluster boundaries get more Hilbert weighting
        cluster_center_dist = jnp.min(jnp.sum((X_perm[:, None, :] - centroids[None, :, :]) ** 2, axis=2), axis=1)
        cluster_weights = jnp.exp(-cluster_center_dist / jnp.median(cluster_center_dist + 1e-8))
        cluster_weights = jnp.clip(cluster_weights, 0.3, 0.8)

        combined_idx = (morton_idx * (1 - cluster_weights) +
                       hilbert_idx * cluster_weights).astype(jnp.uint64)

        order = jnp.argsort(combined_idx)

        # Smooth along curve
        X_sorted = X_low[order]
        X_smooth = smooth_along_curve(X_sorted, window)

        # Restore original order
        inv_order = jnp.zeros(N, dtype=jnp.int32).at[order].set(jnp.arange(N))
        X_restored = X_smooth[inv_order]

        embeddings.append(X_restored)

    # Average ensemble
    embedded = jnp.mean(jnp.stack(embeddings), axis=0)

    return embedded, labels


def random_rotation(key, X):
    """Random orthogonal rotation (JAX-only)."""
    D = X.shape[1]
    M = jax.random.normal(key, (D, D), dtype=X.dtype)
    Q, _ = jnp.linalg.qr(M)
    return X @ Q


def smooth_along_curve(X_sorted, window):
    """Fast convolution smoothing."""
    kernel = jnp.ones((window,), dtype=X_sorted.dtype) / window

    def smooth_dim(x):
        from jax.scipy.signal import convolve
        return convolve(x, kernel, mode='same')

    return jax.vmap(smooth_dim, in_axes=1, out_axes=1)(X_sorted)


def randomized_svd_reduce(X, rank, key):
    """Randomized SVD for dimensionality reduction."""
    N, D = X.shape
    oversample = 8
    n_iter = 2

    # Center the data
    X_centered = X - jnp.mean(X, axis=0)

    l = rank + oversample
    Omega = jax.random.normal(key, (D, l), dtype=X.dtype)

    # Stage A: Find approximate range
    Y = X_centered @ Omega

    # Power iterations
    for _ in range(n_iter):
        Y = X_centered @ (X_centered.T @ Y)

    Q, _ = jnp.linalg.qr(Y)

    # Stage B: Compute SVD on smaller matrix
    B = Q.T @ X_centered
    _, S, Vt = jnp.linalg.svd(B, full_matrices=False)

    # Project data
    X_reduced = X_centered @ Vt.T[:, :rank]

    return X_reduced


# ============================================================
# CLASS INTERFACE
# ============================================================

class DetClustMap(BaseMap):
    """
    JAX-only DetMap with cluster separation.
    """

    def __init__(self, n_components=2, reduced_dims=32, bits=6, window=32,
                 ensemble_size=4, cluster_exaggeration=2.0,
                 use_adaptive_log=True, use_multiscale=True,
                 n_clusters=None, mixing_strategy="adaptive",
                 global_morton_weight=0.3,random_state=42):
        """
        Initialize the cluster-separated DetMap.

        Args:
            n_components: Number of output dimensions
            reduced_dims: Target reduced dimensions before curve embedding
            bits: Bits per dimension for space-filling curve
            window: Smoothing window size
            ensemble_size: Number of random rotations
            cluster_exaggeration: How much to push clusters apart
            use_adaptive_log: Apply log transform to outliers
            use_multiscale: Apply multiscale separation
            n_clusters: Number of clusters (auto if None)
            mixing_strategy: How to combine Morton and Hilbert curves
                Options: "adaptive", "interleaved", "weighted", "ensemble", "alternate"
            global_morton_weight: Weight for Morton in "weighted" strategy
            random_state: Random seed
        """
        self.n_components = n_components
        self.reduced_dims = reduced_dims
        self.bits = bits
        self.window = window
        self.ensemble_size = ensemble_size
        self.cluster_exaggeration = cluster_exaggeration
        self.use_adaptive_log = use_adaptive_log
        self.use_multiscale = use_multiscale
        self.n_clusters = n_clusters
        self.mixing_strategy = mixing_strategy
        self.global_morton_weight = global_morton_weight
        self.random_state = random_state
        self.embedded_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """Just store the training data for reference."""
        self.X_fit_ = self._check_array(X)
        return self

    def transform(self, X):
        """Fit and transform the data."""
        key = jax.random.PRNGKey(self.random_state)

        self.embedded_, self.labels_ = cluster_separated_embedding(
            X,
            reduced_dims=self.reduced_dims,
            bits=self.bits,
            window=self.window,
            ensemble_size=self.ensemble_size,
            cluster_exaggeration=self.cluster_exaggeration,
            use_adaptive_log=self.use_adaptive_log,
            use_multiscale=self.use_multiscale,
            n_clusters=self.n_clusters,
            mixing_strategy=self.mixing_strategy,
            global_morton_weight=self.global_morton_weight,
            key=key
        )

        # Return only requested components
        return np.asarray(self.embedded_[:, :self.n_components])

    def get_embedding(self):
        return self.embedded_

    def get_labels(self):
        return self.labels_




# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Generate test data with clusters
    import numpy as np
    import matplotlib.pyplot as plt

    # Create synthetic clusters
    np.random.seed(42)
    n_points = 1000
    n_clusters = 5

    X_list = []
    for i in range(n_clusters):
        center = np.random.randn(10) * 5
        cluster = center + np.random.randn(n_points // n_clusters, 10) * 0.5
        X_list.append(cluster)

    X = np.vstack(X_list)

    # Apply cluster-separated embedding
    detmap = DetClustMap(
        n_components=2,
        reduced_dims=16,
        bits=5,
        window=16,
        ensemble_size=3,
        cluster_exaggeration=2.0,
        use_adaptive_log=True,
        use_multiscale=True
    )

    X_embedded = detmap.fit_transform(X)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=detmap.get_labels(),
                cmap='tab10', alpha=0.7)
    plt.title("Cluster-Separated Embedding")
    plt.colorbar()

    # Compare with standard PCA
    from sklearn.decomposition import PCA
    X_pca = PCA(n_components=2).fit_transform(X)

    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=detmap.get_labels(),
                cmap='tab10', alpha=0.7)
    plt.title("Standard PCA")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    print(f"Embedding shape: {X_embedded.shape}")
    print(f"Number of clusters detected: {len(np.unique(detmap.get_labels()))}")
