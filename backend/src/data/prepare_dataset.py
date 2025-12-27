"""
Dataset preparation script for AgroVision.

This script converts raw AgriFieldNet GeoTIFF files into PyTorch-ready .npy files.

Usage:
    python -m backend.src.data.prepare_dataset

Output:
    data/processed/
    ├── train_images.npy          # (N_train, 12, 256, 256) float32
    ├── train_masks.npy           # (N_train, 256, 256) int64
    ├── val_images.npy            # (N_val, 12, 256, 256) float32
    ├── val_masks.npy             # (N_val, 256, 256) int64
    ├── normalization_stats.json  # Per-band mean and std
    └── class_map.json            # Class ID to name mapping

    data/splits/
    ├── train_ids.csv             # Training tile IDs
    └── val_ids.csv               # Validation tile IDs
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from tqdm import tqdm

from backend.src.utils.io import (
    get_band_filepath,
    get_label_filepath,
    get_tile_ids_from_source,
    load_band_tiff,
    load_label_tiff,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_valid_tile_ids(config: dict) -> list[str]:
    """
    Get tile IDs that have both source imagery and labels.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    list[str]
        List of valid tile IDs
    """
    source_dir = Path(config["paths"]["data_raw"]) / "source"
    labels_dir = Path(config["paths"]["data_raw"]) / "train_labels"

    # Get all tile IDs from source
    all_tile_ids = get_tile_ids_from_source(source_dir)

    # Filter to only those with labels
    valid_ids = []
    for tile_id in all_tile_ids:
        label_path = get_label_filepath(labels_dir, tile_id, "raster")
        if label_path.exists():
            valid_ids.append(tile_id)

    return valid_ids


def create_train_val_split(
    tile_ids: list[str],
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Split tile IDs into training and validation sets.

    Parameters
    ----------
    tile_ids : list[str]
        List of all tile IDs
    val_split : float
        Fraction for validation (default: 0.2)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[list[str], list[str]]
        (train_ids, val_ids)
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(tile_ids))

    n_val = int(len(tile_ids) * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_ids = [tile_ids[i] for i in train_indices]
    val_ids = [tile_ids[i] for i in val_indices]

    return train_ids, val_ids


def compute_band_statistics(
    tile_ids: list[str],
    config: dict,
    sample_size: Optional[int] = None,
) -> dict:
    """
    Compute per-band mean and standard deviation from training tiles.

    Parameters
    ----------
    tile_ids : list[str]
        List of training tile IDs
    config : dict
        Configuration dictionary
    sample_size : int, optional
        Number of tiles to sample for statistics (None = all tiles)

    Returns
    -------
    dict
        Dictionary with band names as keys, containing 'mean' and 'std'
    """
    source_dir = Path(config["paths"]["data_raw"]) / "source"
    bands = [b["name"] for b in config["bands"]]

    # Sample tiles if requested
    if sample_size and sample_size < len(tile_ids):
        np.random.seed(42)
        sampled_ids = np.random.choice(tile_ids, sample_size, replace=False)
    else:
        sampled_ids = tile_ids

    print(f"Computing statistics from {len(sampled_ids)} tiles...")

    # Accumulate statistics using Welford's online algorithm
    stats = {band: {"count": 0, "mean": 0.0, "M2": 0.0} for band in bands}

    for tile_id in tqdm(sampled_ids, desc="Computing band statistics"):
        for band in bands:
            band_path = get_band_filepath(source_dir, tile_id, band)
            if not band_path.exists():
                continue

            data = load_band_tiff(band_path)
            pixels = data.flatten()

            # Welford's online algorithm for mean and variance
            for x in pixels:
                stats[band]["count"] += 1
                delta = x - stats[band]["mean"]
                stats[band]["mean"] += delta / stats[band]["count"]
                delta2 = x - stats[band]["mean"]
                stats[band]["M2"] += delta * delta2

    # Finalize statistics
    result = {}
    for band in bands:
        count = stats[band]["count"]
        if count > 1:
            variance = stats[band]["M2"] / (count - 1)
            result[band] = {
                "mean": float(stats[band]["mean"]),
                "std": float(np.sqrt(variance)),
                "count": int(count),
            }
        else:
            result[band] = {"mean": 0.0, "std": 1.0, "count": 0}

    return result


def load_tile(
    tile_id: str,
    config: dict,
    norm_stats: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and normalize a single tile.

    Parameters
    ----------
    tile_id : str
        Tile identifier
    config : dict
        Configuration dictionary
    norm_stats : dict
        Normalization statistics per band

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (image, mask) where:
        - image: (12, 256, 256) float32 normalized
        - mask: (256, 256) int64 class IDs
    """
    source_dir = Path(config["paths"]["data_raw"]) / "source"
    labels_dir = Path(config["paths"]["data_raw"]) / "train_labels"
    bands = [b["name"] for b in config["bands"]]
    target_size = (256, 256)

    # Load and normalize bands
    band_data = []
    for band in bands:
        band_path = get_band_filepath(source_dir, tile_id, band)
        if band_path.exists():
            data = load_band_tiff(band_path, target_size)
            # Z-score normalization
            mean = norm_stats[band]["mean"]
            std = norm_stats[band]["std"]
            data = (data - mean) / (std + 1e-6)
        else:
            # If band is missing, use zeros
            data = np.zeros(target_size, dtype=np.float32)
        band_data.append(data)

    image = np.stack(band_data, axis=0).astype(np.float32)  # (12, 256, 256)

    # Load label mask
    label_path = get_label_filepath(labels_dir, tile_id, "raster")
    mask = load_label_tiff(label_path, target_size)  # (256, 256)

    return image, mask


def generate_npy_files(
    tile_ids: list[str],
    config: dict,
    norm_stats: dict,
    output_prefix: str,
) -> tuple[int, dict]:
    """
    Generate .npy files for a set of tiles.

    Parameters
    ----------
    tile_ids : list[str]
        List of tile IDs to process
    config : dict
        Configuration dictionary
    norm_stats : dict
        Normalization statistics
    output_prefix : str
        Prefix for output files ('train' or 'val')

    Returns
    -------
    tuple[int, dict]
        (num_tiles, class_counts) - number of tiles and pixel counts per class
    """
    output_dir = Path(config["paths"]["data_processed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_tiles = len(tile_ids)
    n_bands = len(config["bands"])
    target_size = 256

    # Pre-allocate arrays
    images = np.zeros((n_tiles, n_bands, target_size, target_size), dtype=np.float32)
    masks = np.zeros((n_tiles, target_size, target_size), dtype=np.int64)

    class_counts = {}

    print(f"Processing {n_tiles} tiles for {output_prefix} set...")
    for i, tile_id in enumerate(tqdm(tile_ids, desc=f"Generating {output_prefix}")):
        image, mask = load_tile(tile_id, config, norm_stats)
        images[i] = image
        masks[i] = mask

        # Count class pixels
        unique, counts = np.unique(mask, return_counts=True)
        for cls, cnt in zip(unique, counts):
            cls = int(cls)
            class_counts[cls] = class_counts.get(cls, 0) + int(cnt)

    # Save arrays
    np.save(output_dir / f"{output_prefix}_images.npy", images)
    np.save(output_dir / f"{output_prefix}_masks.npy", masks)

    print(f"Saved {output_prefix}_images.npy: shape {images.shape}")
    print(f"Saved {output_prefix}_masks.npy: shape {masks.shape}")

    return n_tiles, class_counts


def save_splits(
    train_ids: list[str],
    val_ids: list[str],
    config: dict,
) -> None:
    """Save train/val splits to CSV files."""
    splits_dir = Path(config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "train_ids.csv", "w") as f:
        f.write("tile_id\n")
        for tile_id in train_ids:
            f.write(f"{tile_id}\n")

    with open(splits_dir / "val_ids.csv", "w") as f:
        f.write("tile_id\n")
        for tile_id in val_ids:
            f.write(f"{tile_id}\n")

    print(f"Saved splits: {len(train_ids)} train, {len(val_ids)} val")


def save_metadata(
    norm_stats: dict,
    class_counts: dict,
    config: dict,
) -> None:
    """Save normalization stats and class map."""
    output_dir = Path(config["paths"]["data_processed"])

    # Save normalization stats
    with open(output_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    # Create class map from config
    class_map = {
        "classes": {},
        "num_classes": len(config["classes"]),
        "class_counts": class_counts,
    }
    for cls_id, cls_info in config["classes"].items():
        class_map["classes"][str(cls_id)] = {
            "id": int(cls_id),
            "name": cls_info["name"],
            "name_ar": cls_info.get("name_ar", ""),
            "color": cls_info["color"],
        }

    with open(output_dir / "class_map.json", "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"Saved normalization_stats.json and class_map.json")


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("AgroVision Data Preparation Pipeline")
    print("=" * 60)

    # Load config
    config = load_config()
    print(f"\nLoaded config from config/config.yaml")

    # Get valid tile IDs
    print("\nScanning for valid tiles...")
    tile_ids = get_valid_tile_ids(config)
    print(f"Found {len(tile_ids)} tiles with both imagery and labels")

    if len(tile_ids) == 0:
        print("ERROR: No valid tiles found. Please ensure data is downloaded.")
        return

    # Create train/val split
    val_split = config["training"]["val_split"]
    random_seed = config["training"]["random_seed"]
    train_ids, val_ids = create_train_val_split(tile_ids, val_split, random_seed)
    print(f"\nSplit: {len(train_ids)} train, {len(val_ids)} val")

    # Save splits
    save_splits(train_ids, val_ids, config)

    # Compute normalization statistics from training set only
    print("\nComputing normalization statistics...")
    norm_stats = compute_band_statistics(train_ids, config)

    # Generate training set
    print("\n" + "=" * 60)
    n_train, train_class_counts = generate_npy_files(
        train_ids, config, norm_stats, "train"
    )

    # Generate validation set
    print("\n" + "=" * 60)
    n_val, val_class_counts = generate_npy_files(val_ids, config, norm_stats, "val")

    # Combine class counts
    all_class_counts = {}
    for cls, cnt in train_class_counts.items():
        all_class_counts[cls] = all_class_counts.get(cls, 0) + cnt
    for cls, cnt in val_class_counts.items():
        all_class_counts[cls] = all_class_counts.get(cls, 0) + cnt

    # Save metadata
    save_metadata(norm_stats, all_class_counts, config)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {config['paths']['data_processed']}")
    print(f"  - train_images.npy: ({n_train}, 12, 256, 256)")
    print(f"  - train_masks.npy: ({n_train}, 256, 256)")
    print(f"  - val_images.npy: ({n_val}, 12, 256, 256)")
    print(f"  - val_masks.npy: ({n_val}, 256, 256)")
    print(f"  - normalization_stats.json")
    print(f"  - class_map.json")


if __name__ == "__main__":
    main()
