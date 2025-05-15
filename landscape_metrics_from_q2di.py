
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
import numpy as np
import pandas as pd

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

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from dggrid4py import DGGRIDv7
from dggrid4py import igeo7

import igeo7_ext

from igeo7_ext import dggrid_igeo7_q2di_from_cellids

from igeo7_ext import (
    to_parent_series,
    dggrid_igeo7_q2di_from_cellids,
    dggrid_igeo7_grid_cell_polygons_from_cellids,
    dggrid_get_res,
    download_executable
)
# import tqdm
from tqdm import tqdm

import ast
from collections import defaultdict

def calculate_class_proportions(df):
    cell_area_lookup = cell_area_df['average_hexagon_area_m2'].to_dict()

    flat_counts = df.groupby(['resolution', 'landuse_class_flat']).size().reset_index(name='count_flat')
    hier_counts = df.groupby(['resolution', 'landuse_class_hierarchical']).size().reset_index(name='count_hier')

    merged = pd.merge(
        flat_counts,
        hier_counts,
        left_on=['resolution', 'landuse_class_flat'],
        right_on=['resolution', 'landuse_class_hierarchical'],
        how='outer'
    )

    merged['landuse_class_flat'] = merged['landuse_class_flat'].fillna(merged['landuse_class_hierarchical'])
    merged['landuse_class_hierarchical'] = merged['landuse_class_hierarchical'].fillna(merged['landuse_class_flat'])
    merged['count_flat'] = merged['count_flat'].fillna(0)
    merged['count_hier'] = merged['count_hier'].fillna(0)

    def calc_proportions(row):
        res = int(row['resolution'])
        cell_area = cell_area_lookup.get(res, 0)
        total_cells = (df['resolution'] == res).sum()
        total_area = total_cells * cell_area
        return pd.Series({
            'flat': row['count_flat'] * cell_area * 100 / total_area,
            'hier': row['count_hier'] * cell_area * 100 / total_area
        })

    proportions = merged.apply(calc_proportions, axis=1)
    merged['flat_proportion'] = proportions['flat']
    merged['hier_proportion'] = proportions['hier']

    merged = merged[['resolution', 'landuse_class_flat', 'flat_proportion', 'hier_proportion']].round(2)
    merged = merged.rename(columns={'landuse_class_flat': 'code'})

    return merged


def calculate_patch_density_naive(df):
    """
    Calculate patch density (patches per hectare) for each land use class and resolution.
    Returns:
        DataFrame with columns: resolution, code (landuse_class), patch_density_flat, patch_density_hier
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    def compute_density(sub_df, class_column, res):
        indexed_df = sub_df[['I', 'J', class_column]].copy()
        indexed_df['Z7_STRING'] = sub_df.index
        indexed_df = indexed_df.set_index(['I', 'J'])

        z7_dict = indexed_df['Z7_STRING'].to_dict()
        density_records = []

        for majority_class in indexed_df[class_column].dropna().unique():
            class_df = indexed_df[indexed_df[class_column] == majority_class]
            if class_df.empty:
                continue

            name_to_index = {name: idx for idx, name in enumerate(class_df['Z7_STRING'])}

            row_indices, col_indices = [], []
            for t in class_df.itertuples():
                cell_id = z7_dict.get(t.Index)
                neighbors = [
                    (t.Index[0] + 3, t.Index[1] + 1),
                    (t.Index[0] - 2, t.Index[1] - 3),
                    (t.Index[0] + 1, t.Index[1] - 2),
                    (t.Index[0] - 1, t.Index[1] + 2),
                    (t.Index[0] + 2, t.Index[1] + 3),
                    (t.Index[0] - 3, t.Index[1] - 1),
                ]

                for qi, qj in neighbors:
                    neighbor = z7_dict.get((qi, qj))
                    if neighbor in name_to_index:
                        row_indices.append(name_to_index[cell_id])
                        col_indices.append(name_to_index[neighbor])

            connectivity_matrix = csr_matrix(
                (np.ones(len(row_indices)), (row_indices, col_indices)),
                shape=(len(class_df), len(class_df))
            )
            num_patches, _ = connected_components(csgraph=connectivity_matrix, directed=False)
            area_m2 = cell_area_df.loc[res, 'average_hexagon_area_m2']
            total_area_ha = class_df['Z7_STRING'].nunique() * area_m2 / 10000
            patch_density = num_patches / total_area_ha if total_area_ha > 0 else np.nan

            density_records.append((majority_class, patch_density))

        return dict(density_records)

    final_records = []
    for res in sorted(df['resolution'].unique()):
        df_res = df[df['resolution'] == res]
        flat_density = compute_density(df_res, 'landuse_class_flat', res)
        hier_density = compute_density(df_res, 'landuse_class_hierarchical', res)

        all_codes = set(flat_density.keys()).union(hier_density.keys())
        for code in all_codes:
            final_records.append({
                'resolution': res,
                'code': code,
                'patch_density_flat': round(flat_density.get(code, np.nan), 9),
                'patch_density_hier': round(hier_density.get(code, np.nan), 9),
            })

    return pd.DataFrame(final_records)


def calculate_patch_density(df):
    """
    Calculate patch density (patches per hectare) for each land use class and resolution.
    Returns:
        DataFrame with columns: resolution, code (landuse_class), patch_density_flat, patch_density_hier
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    def compute_density(sub_df, class_column, res):
        indexed_df = sub_df[['I', 'J', class_column]].copy()
        indexed_df['Z7_STRING'] = sub_df.index
        indexed_df = indexed_df.set_index(['I', 'J'])

        z7_dict = indexed_df['Z7_STRING'].to_dict()
        density_records = []

        for majority_class in indexed_df[class_column].dropna().unique():
            class_df = indexed_df[indexed_df[class_column] == majority_class]
            if class_df.empty:
                continue

            name_to_index = {name: idx for idx, name in enumerate(class_df['Z7_STRING'])}

            if len(name_to_index) == 1:
                # Only one cell: one patch by default
                num_patches = 1
            else:
                row_indices, col_indices = [], []
                for t in class_df.itertuples():
                    cell_id = z7_dict.get(t.Index)
                    neighbors = [
                        (t.Index[0] + 3, t.Index[1] + 1),
                        (t.Index[0] - 2, t.Index[1] - 3),
                        (t.Index[0] + 1, t.Index[1] - 2),
                        (t.Index[0] - 1, t.Index[1] + 2),
                        (t.Index[0] + 2, t.Index[1] + 3),
                        (t.Index[0] - 3, t.Index[1] - 1),
                    ]
                    for qi, qj in neighbors:
                        neighbor = z7_dict.get((qi, qj))
                        if neighbor in name_to_index:
                            row_indices.append(name_to_index[cell_id])
                            col_indices.append(name_to_index[neighbor])

                if row_indices:
                    connectivity_matrix = csr_matrix(
                        (np.ones(len(row_indices)), (row_indices, col_indices)),
                        shape=(len(class_df), len(class_df))
                    )
                    num_patches, _ = connected_components(csgraph=connectivity_matrix, directed=False)
                else:
                    # No edges, each cell is isolated → each is a separate patch
                    num_patches = len(class_df)

            area_m2 = cell_area_df.loc[res, 'average_hexagon_area_m2']
            total_area_ha = class_df['Z7_STRING'].nunique() * area_m2 / 10000
            patch_density = num_patches / total_area_ha if total_area_ha > 0 else np.nan

            density_records.append((majority_class, patch_density))

        return dict(density_records)

    final_records = []
    for res in sorted(df['resolution'].unique()):
        df_res = df[df['resolution'] == res]
        flat_density = compute_density(df_res, 'landuse_class_flat', res)
        hier_density = compute_density(df_res, 'landuse_class_hierarchical', res)

        all_codes = set(flat_density.keys()).union(hier_density.keys())
        for code in all_codes:
            final_records.append({
                'resolution': res,
                'code': code,
                'patch_density_flat': round(flat_density.get(code, np.nan), 9),
                'patch_density_hier': round(hier_density.get(code, np.nan), 9),
            })

    return pd.DataFrame(final_records)


def calculate_pladj(df):
    """
    Calculate PLADJ (Percentage of Like Adjacencies) per land use class and resolution
    for both flat and hierarchical land use classifications.

    Parameters:
        df (pd.DataFrame): Must include columns I, J, landuse_class_flat, landuse_class_hierarchical,
                           and index as Z7_STRING

    Returns:
        pd.DataFrame: DataFrame with columns: resolution, code (land use class), pladj_flat, pladj_hier
    """
    all_results = []

    for res in sorted(df['resolution'].unique()):
        df_res = df[df['resolution'] == res]

        def compute_pladj(indexed_df, class_col):
            value_dict = indexed_df[class_col].to_dict()
            z7_dict = indexed_df['Z7_STRING'].to_dict()

            like_adjacencies_count = defaultdict(int)
            total_adjacencies_count = defaultdict(int)

            for t in tqdm(indexed_df.itertuples(), desc=f"PLADJ {class_col} res {res}", leave=False):
                cell_id = z7_dict.get(t.Index)
                value_centre = getattr(t, class_col)

                neighbors = [
                    (t.Index[0] + 3, t.Index[1] + 1),
                    (t.Index[0] - 2, t.Index[1] - 3),
                    (t.Index[0] + 1, t.Index[1] - 2),
                    (t.Index[0] - 1, t.Index[1] + 2),
                    (t.Index[0] + 2, t.Index[1] + 3),
                    (t.Index[0] - 3, t.Index[1] - 1),
                ]

                for qi, qj in neighbors:
                    neighbour_class = value_dict.get((qi, qj), np.nan)
                    total_adjacencies_count[value_centre] += 1
                    if value_centre == neighbour_class:
                        like_adjacencies_count[value_centre] += 1

            results = {}
            for patch_type, like_count in like_adjacencies_count.items():
                total_count = total_adjacencies_count[patch_type]
                pladj = (like_count / total_count) * 100 if total_count > 0 else 0
                results[patch_type] = round(pladj, 2)

            return results

        # Create indexed dataframe for both calculations
        base_cols = ['I', 'J', 'landuse_class_flat', 'landuse_class_hierarchical']
        indexed_df = df_res[base_cols].copy()
        indexed_df['Z7_STRING'] = df_res.index
        indexed_df = indexed_df.set_index(['I', 'J'])

        flat_result = compute_pladj(indexed_df, 'landuse_class_flat')
        hier_result = compute_pladj(indexed_df, 'landuse_class_hierarchical')

        all_codes = set(flat_result.keys()).union(hier_result.keys())
        for code in all_codes:
            all_results.append({
                'resolution': res,
                'code': code,
                'pladj_flat': flat_result.get(code, np.nan),
                'pladj_hier': hier_result.get(code, np.nan)
            })

    return pd.DataFrame(all_results)


def calculate_shdi(df):
    grouped_flat = df.groupby(['resolution', 'landuse_class_flat']).size().reset_index(name='count_flat')
    grouped_hier = df.groupby(['resolution', 'landuse_class_hierarchical']).size().reset_index(name='count_hier')

    grouped_flat['area_flat'] = grouped_flat.apply(
        lambda row: row['count_flat'] * cell_area_df.loc[row['resolution'], 'average_hexagon_area_m2'], axis=1
    )
    grouped_hier['area_hier'] = grouped_hier.apply(
        lambda row: row['count_hier'] * cell_area_df.loc[row['resolution'], 'average_hexagon_area_m2'], axis=1
    )

    total_area_flat = grouped_flat.groupby('resolution')['area_flat'].sum().reset_index(name='total_area_flat')
    total_area_hier = grouped_hier.groupby('resolution')['area_hier'].sum().reset_index(name='total_area_hier')

    grouped_flat = pd.merge(grouped_flat, total_area_flat, on='resolution')
    grouped_hier = pd.merge(grouped_hier, total_area_hier, on='resolution')

    grouped_flat['p_i_flat'] = grouped_flat['area_flat'] / grouped_flat['total_area_flat']
    grouped_hier['p_i_hier'] = grouped_hier['area_hier'] / grouped_hier['total_area_hier']

    grouped_flat['shdi_component_flat'] = -grouped_flat['p_i_flat'] * np.log(grouped_flat['p_i_flat'])
    grouped_hier['shdi_component_hier'] = -grouped_hier['p_i_hier'] * np.log(grouped_hier['p_i_hier'])

    shdi_flat = grouped_flat.groupby('resolution')['shdi_component_flat'].sum().reset_index()
    shdi_hier = grouped_hier.groupby('resolution')['shdi_component_hier'].sum().reset_index()

    shdi_df = pd.merge(shdi_flat, shdi_hier, on='resolution')
    shdi_df.columns = ['resolution', 'shdi_flat', 'shdi_hier']
    shdi_df['shdi_flat'] = shdi_df['shdi_flat'].round(4)
    shdi_df['shdi_hier'] = shdi_df['shdi_hier'].round(4)

    return shdi_df


if __name__ == '__main__':
    
    # Path to folder containing tile files
    input_folder = "output/"
    output_folder = "output/metrics/"
    
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

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all resolutions up to 14
    all_res_df = dggrid_get_res(dggrid_instance, "ISEA7H", max_res=14)
    
    # Filter for resolution 7 to 14 and keep resolution as index
    cell_area_df = all_res_df.loc[7:14].copy()
    cell_area_df.index.name = 'isea7h_resolution'
    
    # Get all parquet files matching pattern
    tile_files = [f for f in os.listdir(input_folder) if f.endswith("_full_landuse_q2di.parquet")]
    
    # Loop over tile files
    for filename in tqdm(tile_files, desc="Processing tiles"):
        tile_path = os.path.join(input_folder, filename)
    
        # Extract tile_id from filename
        tile_id = filename.split("_")[1]
    
        # Read the tile parquet
        df = pd.read_parquet(tile_path)
    
        # Skip tile if any res=14 cells have landuse_class == 12
        if ((df['resolution'] == 14) & ((df['landuse_class_flat'] == 12) | (df['landuse_class_hierarchical'] == 12))).any():
            print(f"Skipping tile {tile_id} due to foreign country class (12) at resolution 14")
            continue
    
        # Calculate individual metrics
        proportions_df = calculate_class_proportions(df)
        patch_density_df = calculate_patch_density(df)
        pladj_df = calculate_pladj(df)
        shdi_df = calculate_shdi(df)
    
        # Merge all metrics into a single DataFrame
        all_metrics = proportions_df.merge(patch_density_df, on=['resolution', 'code'], how='outer')
        all_metrics = all_metrics.merge(pladj_df, on=['resolution', 'code'], how='outer')
        all_metrics = all_metrics.merge(shdi_df, on='resolution', how='left')
        all_metrics['tile_id'] = tile_id  # add tile ID for later identification
    
        # Save final result
        output_path = os.path.join(output_folder, f"tile_{tile_id}_metrics.parquet")
        all_metrics.to_parquet(output_path, compression='snappy')


    

