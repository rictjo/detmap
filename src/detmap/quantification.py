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

import numpy as np
import pandas as pd
from .special import randomized_pca_jax , rankdata_jax , local_pca_jax
#
# TAKEN FROM : https://github.com/richardtjornhammar/impetuous/blob/master/src/impetuous/quantification.py as of impetuous version 0.97.1 (late spring 2022)
def multivariate_aligned_pca_legacy ( analytes_df , journal_df ,
                sample_label = 'Sample ID', align_to = 'Modulating group' , n_components=None ,
                add_labels = ['Additional information'] , e2s=None , color_lookup=None , ispec=None ) :

    # SAIGA PROJECTIONS RICHARD TJÖRNHAMMAR
    what                = align_to
    analytes_df         = analytes_df.loc[:,journal_df.columns.values]
    dict_actual         = { sa:di for sa,di in zip(*journal_df.loc[[sample_label,what],:].values) }
    sample_infos        = []
    if not add_labels is None :
        for label in add_labels :
            sample_infos.append( tuple( (label, { sa:cl for cl,sa in zip(*journal_df.loc[[label,sample_label]].values) } ) ) )
    #
    N_P = n_components
    if n_components is None:
        N_P = np.min(np.shape(analytes_df))
    scores,weights,nidx,ncol = local_pca_jax( analytes_df.copy() , ndims = N_P )
    #
    pcaw_df = pd.DataFrame(weights , columns=['PCA '+str(i+1) for i in range(N_P) ] , index=ncol )
    pcas_df = pd.DataFrame( scores , columns=['PCA '+str(i+1) for i in range(N_P) ] , index=nidx )

    corr_r = rankdata_jax(pcas_df.T.apply(lambda x:np.sum(x**2)).values)/len(pcas_df.index)

    pcaw_df .loc[:,what] = [ dict_actual[s] for s in pcaw_df.index.values ]
    projection_df       = pcaw_df.groupby(what).mean() #.apply(np.mean)
    projection_df       = ( projection_df.T / np.sqrt(np.sum(projection_df.values**2,1)) ).T
    projected_df        = pd.DataFrame( np.dot(projection_df,pcas_df.T), index=projection_df.index, columns=pcas_df.index )
    owners  = projected_df.index.values[projected_df.apply(np.abs).apply(np.argmax).values]

    try :
        if ispec is None :
            ispec = int( len(projection_df)>1 )
        specificity = projected_df.apply(np.abs).apply(lambda x:compositional_analysis(x)[ispec] ).values
        pcas_df.loc[:,'Spec,' + {0:'beta',1:'tau',2:'gini',3:'geni'}[ ispec ] ] = specificity
    except NameError:
        # compositional_analysis not available
        pass

    pcas_df .loc[:,'Owner'] = owners
    pcaw_df = pcaw_df.rename( columns={what:'Owner'} )
    pcas_df.loc[:,'Corr,r'] = corr_r

    if not color_lookup is None :
        pcas_df.loc[:, 'Color'] = [ color_lookup[o] for o in pcas_df.loc[:,'Owner'] ]
        pcaw_df.loc[:, 'Color'] = [ color_lookup[o] for o in pcaw_df.loc[:,'Owner'] ]

    if not e2s is None :
        pcas_df.loc[:,'Symbol'] = [ (e2s[g] if not 'nan' in str(e2s[g]).lower() else g) if g in e2s else g for g in pcas_df.index.values ]
    #
    if len( sample_infos ) > 0 :
        for label_tuple in sample_infos :
            label = label_tuple[0]
            sample_lookup = label_tuple[1]
            if not sample_lookup is None :
                pcaw_df.loc[ :, label ] = [ sample_lookup[s] for s in pcaw_df.index.values ]
    return ( pcas_df , pcaw_df )


def multivariate_aligned_pca(analytes_df, journal_df=None,
                             sample_label='Sample ID', align_to='Modulating group',
                             n_components=None, add_labels=None,
                             e2s=None, color_lookup=None, ispec=None):
    """
    Perform multivariate aligned PCA with improved efficiency.

    Parameters
    ----------
    analytes_df : pd.DataFrame
        Data matrix with samples as columns, features as index
    journal_df : pd.DataFrame, optional
        Metadata matrix with samples as columns, annotations as index.
        If None, uses column names as default grouping
    sample_label : str
        Row in journal_df containing sample identifiers
    align_to : str
        Row in journal_df containing grouping variable
    n_components : int, optional
        Number of PCA components
    add_labels : list, optional
        Additional metadata rows to include
    e2s : dict, optional
        Mapping for feature symbols
    color_lookup : dict, optional
        Color mapping for groups
    ispec : int, optional
        Specificity index (0:beta, 1:tau, 2:gini, 3:geni)
    """
    # Handle journal_df = None case - use column names as default grouping
    if journal_df is None:
        # Create a journal with columns as the actual sample names
        journal_df = pd.DataFrame(
            index=[sample_label, align_to],
            columns=analytes_df.columns
        )
        # Fill with sample names
        journal_df.loc[sample_label] = analytes_df.columns.copy()
        journal_df.loc[align_to] = analytes_df.columns.copy()
        analytes_df.columns = ['cid'+str(i) for i in range(len(analytes_df.columns))]
        journal_df.columns  = analytes_df.columns
        journal_df.loc[sample_label] = analytes_df.columns
        add_labels = []  # No additional labels when using defaults

    # Remove mem explision due to bad user input
    analytes_df = analytes_df.loc[:, ~analytes_df.columns.duplicated(keep='first')]
    journal_df  = journal_df.loc[:, ~journal_df.columns.duplicated(keep='first')]
    common_cols = sorted( set(journal_df.columns.values.tolist()) & set(analytes_df.columns.values.tolist()) )
    analytes_df = analytes_df.loc[:, common_cols]
    journal_df  = journal_df.loc[:, common_cols]

    # Build dictionary for align_to mapping (vectorized)
    align_dict = dict(zip(journal_df.loc[sample_label],
                         journal_df.loc[align_to]))

    # Process additional labels (more efficient comprehension)
    sample_infos = []
    if add_labels:
        sample_infos = [(label, dict(zip(journal_df.loc[sample_label],
                                        journal_df.loc[label])))
                       for label in add_labels if label in journal_df.index]

    # Determine number of components
    n_components = n_components or min(analytes_df.shape)

    # Perform local PCA (consider using randomized_pca_jax for speed)
    scores, weights, nidx, ncol = local_pca_jax(analytes_df.copy(),
                                                ndims=n_components)

    # Create DataFrames (more efficient construction)
    pc_cols = [f'PCA {i+1}' for i in range(n_components)]
    pcas_df = pd.DataFrame(scores, columns=pc_cols, index=nidx)
    pcaw_df = pd.DataFrame(weights, columns=pc_cols, index=ncol)

    # Calculate correlation ranks (vectorized)
    corr_r = rankdata_jax((pcas_df ** 2).sum(axis=1).values) / len(pcas_df.index)

    # Add owner mapping (more efficient)
    pcaw_df['Owner'] = [align_dict[s] for s in pcaw_df.index]

    # Group-wise averaging (vectorized)
    projection_df = pcaw_df.groupby('Owner')[pc_cols].mean()

    # Normalize rows (vectorized)
    projection_df = projection_df.div(np.sqrt((projection_df ** 2).sum(axis=1)),
                                     axis=0)

    # Project samples onto group loadings
    projected_df = pd.DataFrame(np.dot(projection_df.values, pcas_df[pc_cols].T),
                               index=projection_df.index,
                               columns=pcas_df.index)

    # owners  = projected_df.index.values[projected_df.apply(np.abs).apply(np.argmax).values]
    # Assign owners based on maximum absolute projection (vectorized)
    owners = projection_df.index[projected_df.abs().values.argmax(axis=0)]
    pcas_df['Owner'] = owners.values

    # Add specificity (if function exists)
    if ispec is None:
        ispec = int(len(projection_df) > 1)

    # Try to compute specificity, handle missing function gracefully
    try:
        specificity = projected_df.abs().apply(
            lambda x: compositional_analysis(x)[ispec], axis=1
        ).values
        spec_key = {0: 'beta', 1: 'tau', 2: 'gini', 3: 'geni'}.get(ispec, 'beta')
        pcas_df[f'Spec,{spec_key}'] = specificity
    except NameError:
        # compositional_analysis not available
        pass

    pcas_df['Corr,r'] = corr_r

    # Add colors if mapping provided
    if color_lookup:
        pcas_df['Color'] = pcas_df['Owner'].map(color_lookup)
        pcaw_df['Color'] = pcaw_df['Owner'].map(color_lookup)

    # Add symbols if mapping provided
    if e2s:
        pcas_df['Symbol'] = pcas_df.index.to_series().map(
            lambda g: e2s.get(g, g) if g in e2s and 'nan' not in str(e2s[g]).lower() else g
        )

    # Add additional sample information
    for label, sample_dict in sample_infos:
        if sample_dict:
            pcaw_df[label] = pcaw_df.index.map(sample_dict)

    return pcas_df, pcaw_df
