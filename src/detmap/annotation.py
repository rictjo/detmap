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
from .special import randomized_pca_jax , rankdata_jax , local_pca_jax
#
# TAKEN FROM : https://github.com/richardtjornhammar/impetuous/blob/master/src/impetuous/quantification.py as of impetuous version 0.97.1
def multivariate_aligned_pca ( analytes_df , journal_df ,
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
    if ispec is None :
        ispec = int( len(projection_df)>1 )
    specificity = projected_df.apply(np.abs).apply(lambda x:compositional_analysis(x)[ispec] ).values

    pcas_df .loc[:,'Owner'] = owners
    pcaw_df = pcaw_df.rename( columns={what:'Owner'} )
    pcas_df.loc[:,'Corr,r'] = corr_r
    pcas_df.loc[:,'Spec,' + {0:'beta',1:'tau',2:'gini',3:'geni'}[ ispec ] ] = specificity

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
