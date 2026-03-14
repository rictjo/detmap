"""Deterministic Multi-scale Ensemble Manifold Embedding."""

# Version
from ._version import __version__

# Main class
from .drotmap import DetMap

# Public API exports - what users can import directly
__all__ = [
    "DetMap",
    "__version__",
]

# Optional: expose key functions at top level
from .quantification import multivariate_aligned_pca

# Optional: expose special functions if useful to users
from .special import randomized_pca_jax, rankdata_jax, local_pca_jax
