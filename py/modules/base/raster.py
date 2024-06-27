"""Raster processing tools"""

import numpy as np
import rasterio as rio
import warnings
from pathlib import Path
from typing import Tuple
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_bounds

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
Bbox = Tuple[float, float, float, float]

def reproject_raster(raster_in: Path, raster_out: Path, dst_crs: str):
    """Reproject GeoTiff raster using rasterio. Based on:
    https://rasterio.readthedocs.io/en/stable/topics/reproject.html
    """
    with rio.open(raster_in) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
    
        with rio.open(raster_out, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def georeference_raster(
        raster_in: Path, bbox: Bbox, crs_out: str,
        raster_out: Path = None, height_offset: int = None, 
        width_offset: int = None, width_mod: int = None, 
        height_mod: int = None):
    """Convert regular png/tif to georeferenced raster by applying bounds to the image."""
    if raster_out is None:
        raster_out = raster_in 
    if width_offset is None:
        width_offset = 0
    if height_offset is None:
        height_offset = 0
    if width_mod is None:
        width_mod = 0
    if height_mod is None:
        height_mod = 0
    bands = [1, 2, 3]
    west, south, east, north = bbox
    with rio.open(raster_in, 'r') as ds:
        data = ds.read(bands)
        transform = from_bounds(
            west-width_offset, south-height_offset, 
            east+width_offset, north+height_offset, 
            ds.width+width_mod, ds.height+height_mod)

    with rio.open(
            raster_out, 'w', driver='GTiff',
            width=ds.width, height=ds.height,
            count=3, dtype=data.dtype, nodata=0,
            transform=transform, crs=crs_out) as dst:
        dst.write(data, indexes=bands)
