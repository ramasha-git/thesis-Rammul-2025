# coding: utf-8

# # Raster GeoTiff (COG) to DGGS (IGEO7)
# 
# Required Python packages (e.g. Colab):
# 
# - requests
# - pandas
# - numpy
# - shapely
# - geopandas
# - rasterio
# - pyarrow
# - tqdm
# - dggrid4py
# 
# For a `micromamba` / `miniconda` enviroment we should install via conda/micromamba

# In[1]:


# could do pip install if imports don't work
# dggrid4py always via pip (not on conda-forge)
# !pip install dggrid4py


# In[2]:


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

from tqdm import tqdm

from dggrid4py import DGGRIDv7
from dggrid4py import igeo7


import igeo7_ext

from igeo7_ext import dggrid_get_res

from igeo7_ext import suggest_window_blocks_per_chunk
# def suggest_window_blocks_per_chunk(rs_src, mem_use_mb):

from igeo7_ext import extract_windows_with_bounds
# def extract_windows_with_bounds(raster_path, window_blocks_per_chunk=None, mem_use_mb=500):


# In[11]:


from igeo7_ext import __haversine
# def __haversine(lon1, lat1, lon2, lat2):

from igeo7_ext import get_crs_info
# def get_crs_info(crs_wkt):

from igeo7_ext import projected_distance
# def projected_distance(east1, north1, east2, north2):

from igeo7_ext import get_raster_pixel_edge_len
# def get_raster_pixel_edge_len(rs_src, adjust_latitudes, pix_size_factor):

from igeo7_ext import propose_dggs_level_for_pixel_length
# def propose_dggs_level_for_pixel_length(dggrid_instance, pixel_edge_len, pix_size_factor, dggrid_dggs="ISEA7H", max_res=16):


# In[12]:


from igeo7_ext import create_geopoints_for_window
# def create_geopoints_for_window(full_transform, window, data_array, crs_ref="EPSG:4326"):


# ## Initialisation of tools and functions
# 
# - setup DGGRID executable
# - load needed library functions for conversion

# In[5]:

# Use the function to download the executable
def get_dggrid():
    url = "https://storage.googleapis.com/geo-assets/dggs-dev/dggrid"
    dggrid_exec = '/Users/akmoch/dev/build/geolynx/ujaval_climate_trends/bin/dggrid-linux-x64'
    
    from igeo7_ext import download_executable
    # def download_executable(url, folder="./"):
    
    if not os.path.isfile(dggrid_exec) and not os.access(dggrid_exec, os.X_OK):
        dggrid_exec = download_executable(url, "./bin")
        print(f"Downloaded and made executable: {dggrid_exec}")
    else:
        print(f"Executable already available: {dggrid_exec}")

# we need the DGGRID executable
# For a `micromamba` / `miniconda` enviroment we can install via conda/micromamba (conda install -c conda-forge dggrid)
# if not in a conda environment, skip to next cell

if __name__ == '__main__':
    
    tile_fn = 'tile_4434.tif'

    print(sys.argv, len(sys.argv))

    if len(sys.argv) == 2 and sys.argv[1].endswith('.tif'):
        tile_fn = sys.argv[1]
        

    dggrid_exec = 'dggrid'
    # raster_path = os.path.expanduser("dem/merit_dem_pori_cog.tif")
    raster_path = '/vsicurl/' + 'https://storage.googleapis.com/geo-assets/grid_tiles/landscape_dggs/' + tile_fn

    
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



# we need to download DGGRID executable (only works on Linux, colab should work)
# For a `micromamba` / `miniconda` enviroment we can install via conda/micromamba (conda install -c conda-forge dggrid)



    print(dggrid_exec)
    
    
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


# In[9]:


# def dggrid_get_res(dggrid_instance, dggrid_dggs="ISEA7H", max_res=16):

    print(dggrid_get_res(dggrid_instance, "ISEA7H", 16))


# In[10]:


# ## Start of the workflow
# 

# ### 1) initial calibration of batch-size
# 
# - set the `raster_path` to the COG GeoTiff file
# - no need to change `mem_use_mb`, `window_blocks_per_chunk`, `pix_size_factor` or other parameters unless otherwise explicitely desired



# !gdal_translate -f COG dem/merit_dem_pori.tif dem/merit_dem_pori_cog.tif


# - for each poly in mapsheets
# -   extract the raster part -> should convert to COG and 4326

# In[15]:


    mem_use_mb = 100
    window_blocks_per_chunk = None
    pix_size_factor=1   # oversampling factor to keep extra detail (between 1 same-ish to 3 very detailed/lots duplication)
    dggrid_dggs_ref="ISEA7H" # ISEA7H and IGEO7 have same grid configuration, but for resolution testing DGGRID can only use ISEA7H, but for cell-ids in the workflow we want IGEO7/Z7 cell ids
    dggs_type="IGEO7"
    max_res=16
    pixel_edge_len = -1
    src_crs_ref = "EPSG:4326"
    src_epsg = 4326
    
    with rasterio.open(raster_path) as src:
        # Determine chunk size if not provided
        print("##############################################")
        print("############## calibrating initial conversion parameters to balance workload ##########")
        print("---- raster metadata ----")
        print(src.meta)
        print("---- raster crs ----")
        print(src.meta['crs'])
        src_crs_ref = src.meta['crs']
        src_epsg = src_crs_ref.to_epsg()
        print("---- raster blocks ----")
        if window_blocks_per_chunk is None:
            window_blocks_per_chunk = suggest_window_blocks_per_chunk(src, mem_use_mb)
        print("---- guess approx. pixel side length ----")
        pixel_edge_len = get_raster_pixel_edge_len(src, adjust_latitudes=True, pix_size_factor=2)
        print(pixel_edge_len)
        print("---- proposising dggs grid resolution ----")
        propose_dggs_info_dict = propose_dggs_level_for_pixel_length(dggrid_instance, pixel_edge_len, pix_size_factor=pix_size_factor, dggrid_dggs=dggrid_dggs_ref, max_res=max_res)
        dg_level = propose_dggs_info_dict['resolution']
        print(f"Proposed DGGS resolution: {dg_level}")
    
    
# In[16]:


    # manual overwrite for testing, only keep active if sure
    dg_level = 14
    # window_blocks_per_chunk=2
    
    
    # In[17]:
    
    
    # review batch sizes/place (raster data block windows)
    bounds_list = []
    
    for window, bounds, data, transform in extract_windows_with_bounds(raster_path, window_blocks_per_chunk=window_blocks_per_chunk, mem_use_mb=mem_use_mb):
        # bounds contains (minx, miny, maxx, maxy) in geographic coordinates
        # print(bounds)
        # print(data.shape)
        bbox = bbox = box(*bounds)
        bounds_list.append(bbox)
    
    expected_window_steps = len(bounds_list)
    print(bounds)
    # gpd.GeoDataFrame({'geometry': bounds_list}, crs=src_crs_ref).explore()
    

# ### Batch conversion of window blocks to separate DGGS Parquet
# 
# - make a data_folder into where we export all the separate parquet files for this one raster source file

# In[19]:

    
    main_output_folder = 'output'
    parquet_subfolder = os.path.basename(raster_path).replace(".tiff", "").replace(".tif", "") + "_parquet_out"
    target_folder = os.path.join(main_output_folder, parquet_subfolder)
    param_parquet_compression = "snappy"
    parquet_filename_template = 'data-{0}.parquet'.format
    
    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    
    
    # In[20]:
    
    
    print(f"Expecting {expected_window_steps} conversion steps/output parquet files in: {target_folder}")


# In[21]:


# describe the target data variable
    var_desc = {
        'metadata' : 'ETAK landuse base classes.',
        'var_name': 'landuse_class',  # no spaces or special characters
        'var_type': 'categorical',   # scalar or categorical
        'src_raster_origin': raster_path,
        'src_raster_crs': src_epsg,
        'dggs_type': dggs_type,
        'sampling_dggs_level': dg_level,
        'sampling_cls_m': propose_dggs_info_dict['cls_m'],
        'sampling_average_hexagon_area_m2': propose_dggs_info_dict['average_hexagon_area_m2'],
        'sampling_pixel_edge_len': pixel_edge_len,
        'param_window_blocks_per_chunk': window_blocks_per_chunk,
        'param_mem_use_mb': mem_use_mb,
        'param_pix_size_factor': pix_size_factor,
        'param_max_res': max_res,
        'param_parquet_compression': param_parquet_compression,
        'param_parquet_filename_template': parquet_filename_template('XXX')
    }
    print(f"Folowing metdata in {target_folder}/metadata.json")
    
    with open(os.path.join(target_folder, "metadata.json"), 'w') as fh:
        json.dump(var_desc, fh)
        
    print(var_desc)


# In[22]:


    with open(os.path.join(target_folder, "metadata.json"), 'r') as fh:
        var_desc = json.load(fh)
        
    print(var_desc)
    
    
    for idx, (window, bounds, window_data, full_transform) in tqdm(enumerate(extract_windows_with_bounds(raster_path, window_blocks_per_chunk=window_blocks_per_chunk, mem_use_mb=mem_use_mb))):
        # bounds contains (minx, miny, maxx, maxy) in geographic coordinates
        # data contains the numpy array for this window
        # restore the local transform for this window
        window_transform = rasterio.windows.transform(window, full_transform)
        print(f"Window transform: {window_transform}")
        
        # Make a bbox from the bounds of this window
        print(f"Window bounds (left, bottom, right, top): {bounds}")
        bbox = box(*bounds)
        if src_epsg != 4326:
            bbox = gpd.GeoSeries([bbox], crs=src_crs_ref).to_crs(4326).loc[0]
            print(bbox)
    
        geodf_points_in_crs = create_geopoints_for_window(full_transform, window, window_data, crs_ref=src_crs_ref)
        # gdf_coded_centroids = dggrid_instance.cells_for_geo_points(geodf_points_wgs84=geodf_points_wgs84, cell_ids_only=True, dggs_type='IGEO7', resolution=dg_level, output_address_type='Z7_STRING')
        geodf_points_in_crs.sindex
        
        # if we oversample at (much) higher resolution as the source data, we don't need polygons
        # gdf_grid = dggrid_instance.grid_cell_polygons_for_extent('IGEO7', dg_level, clip_geom=bbox, output_address_type='Z7_STRING')
        gdf_hexpoints = dggrid_instance.grid_cell_centroids_for_extent('IGEO7', dg_level, clip_geom=bbox, output_address_type='Z7_STRING')
        gdf_hexpoints.crs = 4326
        if src_epsg != 4326:
            gdf_hexpoints = gdf_hexpoints.to_crs(src_crs_ref)
        gdf_hexpoints.sindex
    
        joined_hexpoints = gpd.sjoin_nearest(gdf_hexpoints[['name', 'geometry']], geodf_points_in_crs[['data', 'geometry']], how='left', distance_col='distance')
        df = joined_hexpoints[['name','data']].rename(columns={'name':'cellids', 'data': var_desc['var_name']}).set_index('cellids', drop=True).dropna()
        if df.index.size > 0:
            # Convert DataFrame to PyArrow Table
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                os.path.join(target_folder, parquet_filename_template(idx)),
                compression=param_parquet_compression
            )
    
    dfs = []
    tp_prefix, tp_suffix = parquet_filename_template("XXXX").split("XXXX")
    pq_name = os.path.basename(target_folder).replace("_parquet_out", ".parquet")

    for fn in list( filter( lambda s: s.startswith(tp_prefix) and s.endswith(tp_suffix), os.listdir(target_folder) ) ):
        print(os.path.join(target_folder, fn))
        df = pd.read_parquet(os.path.join(target_folder, fn))
        dfs.append(df)

    df = pd.concat(dfs)
    if df.index.size > 0:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, os.path.join(main_output_folder, pq_name), compression="snappy")
