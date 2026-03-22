"""Map implementations for detmap."""

# Make maps available at detmap.maps.*
from .dmap import DMap
from .drotmap import DetMap  # Keep DetMap as primary
from .dhiemap import DhieMap
from .detsfcmap import DetSFCMap
from .clustdetmap import DetClustMap
from .bitmap import OptimalHybridMap, BitInterleavedClusterMap, EnhancedOptimalHybridMap, NonlinearHybridMap

__all__ = ['DMap', 'DetMap', 'DhieMap', 'DetSFCMap', 'DetClustMap',
           'OptimalHybridMap','BitInterleavedClusterMap','EnhancedOptimalHybridMap','NonlinearHybridMap']
