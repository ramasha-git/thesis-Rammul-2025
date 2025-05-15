import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


from pathlib import Path
import os
import io
import sys
import copy
import stat
import shutil
import math
import json

import requests
import pyproj
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import box, mapping, shape
from shapely.geometry import Point, Polygon
from shapely.ops import transform

import rasterio
from rasterio.windows import Window
from affine import Affine

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from dggrid4py import DGGRIDv7
from dggrid4py import igeo7
from tqdm import tqdm

import igeo7_ext

from igeo7_ext import dggrid_igeo7_q2di_from_cellids

from igeo7_ext import to_parent_series

from igeo7_ext import (
    to_parent_series,
    dggrid_igeo7_q2di_from_cellids,
    dggrid_igeo7_grid_cell_polygons_from_cellids,
    dggrid_get_res,
    download_executable
)



def flat_landuse_aggregation(df_raw):
    """
    Perform flat land use aggregation from resolution 14 to 8.
    Each level uses raw resolution 14 data.

    Returns:
        df_flat_all (DataFrame): cellid, resolution, landuse_class_flat
    """
    df_flat = df_raw.copy()
    df_flat['res_14'] = df_flat['cellid']
    flat_results = [df_flat[['cellid', 'resolution', 'landuse_class']]]

    for res in tqdm(range(14, 7, -1), desc='Flat aggregation'):
        res_col = f'res_{res}'
        parent_col = f'res_{res - 1}'

        if parent_col not in df_flat.columns:
            df_flat[parent_col] = to_parent_series(df_flat[res_col])

        grouped = df_flat.groupby(parent_col)['landuse_class'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index()

        grouped.columns = ['cellid', 'landuse_class']
        grouped['resolution'] = res - 1
        flat_results.append(grouped)

    df_flat_all = pd.concat(flat_results, ignore_index=True)
    df_flat_all.rename(columns={'landuse_class': 'landuse_class_flat'}, inplace=True)
    return df_flat_all


def hierarchical_landuse_aggregation(df_raw):
    """
    Perform hierarchical land use aggregation from resolution 14 to 8.
    Each level uses the result of the previous level.

    Returns:
        df_hier_all (DataFrame): cellid, resolution, landuse_class_hierarchical
    """
    df_hier = df_raw.copy()
    df_hier['parent_13'] = to_parent_series(pd.Series(df_hier['cellid']))

    hierarchical_results = [
        df_hier[['cellid', 'resolution', 'landuse_class']].rename(
            columns={'landuse_class': 'landuse_class_hierarchical'}
        )
    ]

    for res in tqdm(range(13, 7, -1), desc='Hierarchical aggregation'):
        prev_df = hierarchical_results[-1]
        parent_col = f'parent_{res}'

        if parent_col not in prev_df.columns:
            prev_df[parent_col] = to_parent_series(pd.Series(df_hier['cellid']) if res == 13 else pd.Series(prev_df['cellid']))

        grouped = prev_df.groupby(prev_df[parent_col])['landuse_class_hierarchical'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        ).reset_index()

        grouped.columns = ['cellid', 'landuse_class_hierarchical']
        grouped['resolution'] = res

        if res > 8:
            df_hier[f'parent_{res - 1}'] = to_parent_series(df_hier[parent_col])

        hierarchical_results.append(grouped)

    df_hier_all = pd.concat(hierarchical_results, ignore_index=True)
    return df_hier_all


def merge_aggregations(df_flat, df_hier, output_path=None):
    """
    Merge flat and hierarchical aggregation results and optionally save to disk.

    Parameters:
        df_flat (DataFrame): flat aggregation result
        df_hier (DataFrame): hierarchical aggregation result
        output_path (str): Optional path to save merged DataFrame as Parquet

    Returns:
        final_df (DataFrame)
    """
    final_df = pd.merge(
        df_flat,
        df_hier,
        on=['cellid', 'resolution'],
        how='outer'
    ).sort_values(['resolution', 'cellid'])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_parquet(output_path, compression='snappy')

    return final_df


def generate_landuse_q2di_table(final_df, dggrid_instance):
    """
    Create a combined DataFrame with cellid, resolution, landuse_class_flat, landuse_class_hierarchical,
    and Q2DI information (Q, I, J) across all resolutions in the input DataFrame.

    Parameters:
        final_df (DataFrame): Must contain 'cellid', 'resolution',
                              'landuse_class_flat', and 'landuse_class_hierarchical'.
        dggrid_instance (DGGRIDv7): Initialized DGGRID instance from dggrid4py.

    Returns:
        DataFrame: Indexed by cellid, with columns:
                   resolution, landuse_class_flat, landuse_class_hierarchical, Q2DI, Q, I, J
    """
    all_results = []

    for res in sorted(final_df['resolution'].unique()):
        df = final_df[final_df['resolution'] == res][['cellid', 'resolution', 'landuse_class_flat', 'landuse_class_hierarchical']].copy()

        # Compute Q2DI values
        q2di_df = dggrid_igeo7_q2di_from_cellids(df['cellid'].values, dggrid_instance)
        df = df.set_index('cellid').join(q2di_df.set_index('Z7_STRING', drop=True))
        df['Q2DI'] = df[['Q', 'I', 'J']].astype(str).agg(' '.join, axis=1)

        all_results.append(df)

    combined_df = pd.concat(all_results).sort_values(['resolution', 'Q', 'I', 'J'])
    return combined_df[['resolution', 'landuse_class_flat', 'landuse_class_hierarchical', 'Q2DI', 'Q', 'I', 'J']]


# we need the DGGRID executable
# For a `micromamba` / `miniconda` enviroment we can install via conda/micromamba (conda install -c conda-forge dggrid)
# if not in a conda environment, skip to next cell

dggrid_exec = 'dggrid'

# if installed with conda/micromamba or compiled locally, we can ask the system for the location
if not os.path.isfile(dggrid_exec) and not os.access(dggrid_exec, os.X_OK):
    if shutil.which("dggrid83"):
        dggrid_exec = shutil.which("dggrid83")
    elif shutil.which("dggrid8"):
        dggrid_exec = shutil.which("dggrid8")
    elif shutil.which("dggrid"):
        dggrid_exec = shutil.which("dggrid")
    else:
        print("No dggrid executable found in path")

    if os.path.isfile(dggrid_exec) and os.access(dggrid_exec, os.X_OK):
        print(f"A DGGRID executable is available at: {dggrid_exec}")
    else:
        print("No usable dggrid executable found in path")
else:
    print(f"Executable already available: {dggrid_exec}")


from dggrid4py import DGGRIDv7
from dggrid4py import igeo7

if not os.path.isfile(dggrid_exec) or not os.access(dggrid_exec, os.X_OK):
    raise ValueError(f"dggrid executable not found or not executable: {dggrid_exec}")

working_dir=os.curdir
capture_logs=True
silent=True
# if self-compiled or installed with conda, we can set performance options `tmp_geo_out_legacy=False`and `has_gdal=True`
tmp_geo_out_legacy=False
has_gdal=True
debug=False

dggrid_instance = DGGRIDv7(
    executable=dggrid_exec, working_dir=working_dir, capture_logs=capture_logs, silent=silent, tmp_geo_out_legacy=tmp_geo_out_legacy, has_gdal=has_gdal, debug=debug
)


from igeo7_ext import dggrid_get_res
# def dggrid_get_res(dggrid_instance, dggrid_dggs="ISEA7H", max_res=16):

dggrid_get_res(dggrid_instance, "ISEA7H", 14)


# Load tile index from your GPKG
tile_index_gdf = gpd.read_file("mapsheets_50k_samesize.gpkg")

# Loop through each row (tile)
for idx, row in tqdm(tile_index_gdf.iterrows(), total=tile_index_gdf.shape[0], desc="Processing tiles"):
    tile_id = row["NR"]

    parquet_url = os.path.join("output", f"tile_{tile_id}_clipped_final_df.parquet")
    output_path = f"output/tile_{tile_id}_full_landuse_q2di.parquet"

    if tile_id in [4434]:
        print(f"skipping {tile_id}, already done")
        continue

    if os.path.exists(output_path):
        print(f"skipping {output_path}, already done")
        continue
        
    if not os.path.exists(parquet_url):
        print(f"skipping {tile_id}, file not ready")
        continue


    print(f"Processing tile NR={tile_id}")

    # Load raw data
    merged_df = pd.read_parquet(parquet_url)
    # Add Q2DI info
    final_df = generate_landuse_q2di_table(merged_df, dggrid_instance)
    final_df.to_parquet(output_path, compression="snappy")

