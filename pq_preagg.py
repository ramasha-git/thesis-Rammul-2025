import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dggrid4py import DGGRIDv7
from dggrid4py import igeo7

import igeo7_ext

from igeo7_ext import dggrid_igeo7_q2di_from_cellids

import pandas as pd
import numpy as np

from igeo7_ext import to_parent_series
import tqdm

import os
import sys

folder = "output"

if __name__ == '__main__':


    tile_fn = 'tile_4434_clipped.parquet'

    print(sys.argv, len(sys.argv))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.parquet'):
        tile_fn = sys.argv[1]
    # Load base resolution 14 data
    df_raw = pd.read_parquet(os.path.join(folder, tile_fn))
    df_raw.sample(3)

    df_raw['cellid'] = df_raw.index
    df_raw['resolution'] = 14
    df_raw = df_raw[['cellid', 'resolution', 'landuse_class']]
     
    # ---------- Flat Aggregation ----------
    df_flat = df_raw.copy()
    df_flat['res_14'] = df_flat['cellid']
     
    flat_results = [df_flat[['cellid', 'resolution', 'landuse_class']]]
     
    for res in tqdm.tqdm(range(14, 7, -1)):
        
        res_col = f'res_{res}'
        parent_col = f'res_{res - 1}'
     
        if parent_col not in df_flat.columns:
            df_flat[parent_col] = to_parent_series(df_flat[res_col])
    
        print(f"res {res} - start agg")
        grouped = df_flat.groupby(parent_col)['landuse_class'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index()
    
        print(f"res {res} - agged")
        grouped.columns = ['cellid', 'landuse_class']
        grouped['resolution'] = res - 1
        df_flat[parent_col] = df_flat[parent_col]  # retain for later
        flat_results.append(grouped)

    df_flat_all = pd.concat(flat_results, ignore_index=True)
    df_flat_all.rename(columns={'landuse_class': 'landuse_class_flat'}, inplace=True)

    df_hier = df_raw.copy()
    df_hier['parent_13'] = to_parent_series(pd.Series(df_hier['cellid']))
    
    hierarchical_results = [df_hier[['cellid', 'resolution', 'landuse_class']].rename(columns={'landuse_class': 'landuse_class_hierarchical'})]

    for res in tqdm.tqdm(range(13, 7, -1)):
        prev_df = hierarchical_results[-1]
        parent_col = f'parent_{res}'
     
        if parent_col not in prev_df.columns:
            prev_df[parent_col] = to_parent_series(pd.Series(df_hier['cellid']) if res == 13 else pd.Series(prev_df['cellid']))
    
        print(f"res {res} - start agg")
        grouped = prev_df.groupby(prev_df[parent_col])['landuse_class_hierarchical'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index()
        
        print(f"res {res} - agged")
        grouped.columns = ['cellid', 'landuse_class_hierarchical']
        grouped['resolution'] = res
    
        # is that even needed?
        if res > 8:
            df_hier[f'parent_{res - 1}'] = to_parent_series(df_hier[parent_col])
       
        hierarchical_results.append(grouped)
     
    df_hier_all = pd.concat(hierarchical_results, ignore_index=True)

    # ---------- Merge Results ----------
    # Merge both by resolution + cellid
    final_df = pd.merge(
        df_flat_all,
        df_hier_all,
        on=['cellid', 'resolution'],
        how='outer'
    ).sort_values(['resolution', 'cellid'])
     
    # Preview
    print(final_df.head(3))

    out_fn = tile_fn.replace(".parquet", "_final_df.parquet")

    final_df.to_parquet(os.path.join(folder, out_fn), compression='snappy')
    


