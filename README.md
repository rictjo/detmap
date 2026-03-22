# DetMap

A deterministic, SFC (space-filling curve)-based, multi-projection, multi-scale, ensemble manifold embedding with high-dimensional support and near-linear scaling

## Intended usage pattern

```
import pandas as pd
from detmap import DetMap,DetSFCMap,DhieMap,DMap
import jax.numpy as jnp

detmap = DetMap(reduced_dims=2)

if True :
    analytes = pd.read_csv('../data/analytes.tsv',sep='\t',index_col=0 )
    analytes .columns = [c.split('.')[0] for c in analytes.columns]
    labels = None
else :
    # data https://zenodo.org/records/7246239/files/data.zip?download=1
    analytes = pd.read_csv('../data/mnist_data.tsv',sep='\t',index_col=0 )
    labels = [str(s) for s in pd.read_csv('../data/mnist_target.tsv',sep='\t',index_col=0 ).values.tolist()]

X_embedded = detmap.fit_transform(jnp.array(analytes.values))
sdf = pd.DataFrame(X_embedded,index=analytes.index,columns=['comp'+str(i) for i in range(X_embedded.shape[1])])

if labels is None :
    from detmap import multivariate_aligned_pca
    scores, loadings = multivariate_aligned_pca(analytes)
    labels = scores['Owner'].values.tolist()

from detmap.visual import plot_colored_points , plot_colored_points_with_hover

plot_colored_points( x = sdf['comp0'].values ,
                     y = sdf['comp1'].values ,
                     labels = labels )

import matplotlib.pyplot as plt
plt.show()
```
