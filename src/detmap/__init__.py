"""Deterministic Multi-scale Ensemble Manifold Embedding."""

from ._version import __version__
import importlib
import warnings

# Map types and their corresponding modules
_MAP_REGISTRY = {
    'DetMap': 'drotmap',      # Default to drotmap for backward compatibility
    'DMap': 'dmap',
    'Dmap': 'dmap',           # Allow case-insensitive variations
    'DrotMap': 'drotmap',
    'Drotmap': 'drotmap',
    'DROTMap': 'drotmap',
    'Dhiemap': 'dhiemap',
    'DhieMap': 'dhiemap',
    'DHIEMap': 'dhiemap',
}

def __getattr__(name):
    """
    Dynamically load map classes when requested.
    
    This allows users to do:
    from detmap import DetMap, DMap, DhieMap
    
    Without loading all modules at import time.
    """
    if name in _MAP_REGISTRY:
        module_name = _MAP_REGISTRY[name]
        try:
            module = importlib.import_module(f'.maps.{module_name}', package=__name__)
            map_class = getattr(module, name)
            return map_class
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Could not load {name} from {module_name}: {e}")
            raise ImportError(f"Map class {name} not available")
    
    # For other attributes (like functions), try importing from quantification
    try:
        from .quantification import multivariate_aligned_pca as _mvpca
        if name == 'multivariate_aligned_pca':
            return _mvpca
    except ImportError:
        pass
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# For static analysis tools and IDE autocomplete
__all__ = [
    "__version__",
    "DetMap",
    "DMap", 
    "Dhiemap",
    "multivariate_aligned_pca",
]

# Optional: Explicitly expose commonly used functions
from .quantification import multivariate_aligned_pca
from .special import randomized_pca_jax, rankdata_jax, local_pca_jax
