"""
I/O utilities for loading GeoTIFF files and resampling bands.

This module provides functions to:
- Load individual Sentinel-2 band GeoTIFFs
- Load crop label masks
- Resample bands to a consistent resolution (256x256)
"""

from pathlib import Path
from typing import Union

import numpy as np
import rasterio
from rasterio.enums import Resampling


def load_band_tiff(
    filepath: Union[str, Path],
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Load a single-band GeoTIFF and resample to target size.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file
    target_size : tuple[int, int]
        Target (height, width) for output array. Default is (256, 256).

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with float32 dtype
    """
    filepath = Path(filepath)

    with rasterio.open(filepath) as src:
        # Check if resampling is needed
        if src.height == target_size[0] and src.width == target_size[1]:
            # No resampling needed
            data = src.read(1).astype(np.float32)
        else:
            # Resample to target size using bilinear interpolation
            data = src.read(
                1,
                out_shape=target_size,
                resampling=Resampling.bilinear,
            ).astype(np.float32)

    return data


def load_label_tiff(
    filepath: Union[str, Path],
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Load a label mask GeoTIFF and resample to target size.

    Uses nearest-neighbor resampling to preserve discrete class values.

    Parameters
    ----------
    filepath : str or Path
        Path to the label GeoTIFF file
    target_size : tuple[int, int]
        Target (height, width) for output array. Default is (256, 256).

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with int64 dtype containing class IDs
    """
    filepath = Path(filepath)

    with rasterio.open(filepath) as src:
        if src.height == target_size[0] and src.width == target_size[1]:
            data = src.read(1).astype(np.int64)
        else:
            # Use nearest neighbor for labels to preserve discrete values
            data = src.read(
                1,
                out_shape=target_size,
                resampling=Resampling.nearest,
            ).astype(np.int64)

    return data


def resample_to_target_size(
    data: np.ndarray,
    target_size: tuple[int, int],
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resample a 2D array to target size.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array
    target_size : tuple[int, int]
        Target (height, width)
    method : str
        Resampling method: 'bilinear' for continuous data, 'nearest' for labels

    Returns
    -------
    np.ndarray
        Resampled 2D array
    """
    from scipy.ndimage import zoom

    if data.shape == target_size:
        return data

    zoom_factors = (target_size[0] / data.shape[0], target_size[1] / data.shape[1])

    if method == "nearest":
        return zoom(data, zoom_factors, order=0)
    else:  # bilinear
        return zoom(data, zoom_factors, order=1)


def get_tile_ids_from_source(source_dir: Union[str, Path]) -> list[str]:
    """
    Extract unique tile IDs from the source directory.

    The data is organized in subdirectories:
    source/ref_agrifieldnet_competition_v1_source_{tile_id}/

    Parameters
    ----------
    source_dir : str or Path
        Path to the source directory containing tile subdirectories

    Returns
    -------
    list[str]
        Sorted list of unique tile IDs
    """
    source_dir = Path(source_dir)
    tile_ids = set()

    # Each tile has its own subdirectory
    for tile_dir in source_dir.iterdir():
        if tile_dir.is_dir() and tile_dir.name.startswith("ref_agrifieldnet"):
            # Extract tile_id from directory name
            # Pattern: ref_agrifieldnet_competition_v1_source_{tile_id}
            parts = tile_dir.name.split("_")
            if len(parts) >= 5:
                tile_id = parts[5]  # tile_id is after "source"
                tile_ids.add(tile_id)

    return sorted(tile_ids)


def get_band_filepath(
    source_dir: Union[str, Path],
    tile_id: str,
    band_name: str,
) -> Path:
    """
    Construct the filepath for a specific band of a tile.

    Data is organized as:
    source/ref_agrifieldnet_competition_v1_source_{tile_id}/{filename}.tif

    Parameters
    ----------
    source_dir : str or Path
        Path to the source directory
    tile_id : str
        Tile identifier (e.g., '001c1')
    band_name : str
        Band name (e.g., 'B02', 'B8A')

    Returns
    -------
    Path
        Full path to the band GeoTIFF file
    """
    source_dir = Path(source_dir)
    # Subdirectory: ref_agrifieldnet_competition_v1_source_{tile_id}
    tile_dir = f"ref_agrifieldnet_competition_v1_source_{tile_id}"
    # Filename: ref_agrifieldnet_competition_v1_source_{tile_id}_{band}_10m.tif
    filename = f"ref_agrifieldnet_competition_v1_source_{tile_id}_{band_name}_10m.tif"
    return source_dir / tile_dir / filename


def get_label_filepath(
    labels_dir: Union[str, Path],
    tile_id: str,
    label_type: str = "raster",
) -> Path:
    """
    Construct the filepath for a label file.

    Parameters
    ----------
    labels_dir : str or Path
        Path to the labels directory
    tile_id : str
        Tile identifier
    label_type : str
        Type of label: 'raster' for crop labels, 'field_ids' for field IDs

    Returns
    -------
    Path
        Full path to the label GeoTIFF file
    """
    labels_dir = Path(labels_dir)

    if label_type == "field_ids":
        filename = f"ref_agrifieldnet_competition_v1_labels_train_{tile_id}_field_ids.tif"
    else:  # raster labels
        filename = f"ref_agrifieldnet_competition_v1_labels_train_{tile_id}.tif"

    return labels_dir / filename
