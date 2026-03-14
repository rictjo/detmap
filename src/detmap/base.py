"""Base classes for all map implementations."""

from abc import ABC, abstractmethod
import numpy as np

class BaseMap(ABC):
    """
    Abstract base class for all map embeddings.
    
    All map implementations should inherit from this class
    to ensure consistent API.
    """
    
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        
    @abstractmethod
    def fit(self, X, y=None):
        """Fit the embedding."""
        pass
    
    @abstractmethod
    def transform(self, X):
        """Transform data to embedding space."""
        pass
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    # Common utility methods that might be shared
    def _check_array(self, X):
        """Validate input array."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected 2D array")
        return X
