# DetMap

A deterministic, SFC (space-filling curve)-based, multi-projection, multi-scale, ensemble manifold embedding with high-dimensional support and near-linear scaling

## Intended usage pattern

```
# Simple imports - they work the same regardless of where classes live
from detmap import DetMap, DMap, DhieMap

# Create instances
detmap = DetMap()
dmap = DMap(n_components=2)
dhiemap = Dhiemap(n_components=5, hierarchy_depth=4)

# Fit and transform
X_embedded = detmap.fit_transform(X)

# Automatic annotation (leave out metadata_df do align with column labels)
from detmap import multivariate_aligned_pca
scores, loadings = multivariate_aligned_pca(data_df, metadata_df)
```
