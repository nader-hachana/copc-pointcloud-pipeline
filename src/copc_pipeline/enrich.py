"""Enrichment: noise filtering, ground surface, height above ground, voxel aggregation.

Every function here works on plain numpy arrays in and out, no file or
network access, so all of it is unit tested with small made up point clouds
instead of the real file.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_cell_id(x: np.ndarray, y: np.ndarray, xmin: float, ymin: float, cell_size: float, nx: int, ny: int) -> np.ndarray:
    """Which flat 2D grid cell each point falls into, as one integer id per point."""
    cx = np.clip(((x - xmin) / cell_size).astype(np.int64), 0, nx - 1)
    cy = np.clip(((y - ymin) / cell_size).astype(np.int64), 0, ny - 1)
    return cy * nx + cx


def filter_noise(withheld: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop points the file's own provider already flagged as unreliable.

    A per-cell statistical outlier check, flagging any point far from its
    cell's median Z, was tried and cut after testing it against the real
    file: this site has genuine vertical structure, ground directly under a
    roof, sometimes inside the very same small cell, and a check like that
    cannot tell "a real second surface" from "noise", it flags whichever
    cluster is smaller either way. One real cell found during testing held
    points at both z=1.4 and z=44.0, ground and roof, and the filter was
    silently discarding the roof points as noise.

    The ground estimate computed next already uses a low percentile rather
    than a raw minimum specifically so it stays robust to the occasional
    genuine stray low point without needing a separate filtering pass.
    """
    keep = ~withheld
    return keep, {"withheld": int((~keep).sum())}


def compute_ground_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cell_size: float,
    percentile: float = 10.0,
) -> tuple[np.ndarray, int, int]:
    """A low percentile of Z per cell, a robust stand-in for ground elevation.

    A low percentile is used instead of the minimum on purpose: the minimum
    is exactly one noisy point away from being wrong, a percentile needs
    several points to agree before it moves.

    Cells with no points at all get filled from their nearest cell that does
    have one, since a tile's ground estimate should not have holes in it.
    """
    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)

    ground_z = np.full(nx * ny, np.nan)
    for cid in np.unique(cell_id):
        ground_z[cid] = np.percentile(z[cell_id == cid], percentile)

    ground_z = ground_z.reshape(ny, nx)
    empty = np.isnan(ground_z)
    if empty.any() and not empty.all():
        _, nearest = distance_transform_edt(empty, return_indices=True)
        ground_z = ground_z[tuple(nearest)]

    return ground_z.reshape(-1), nx, ny


def height_above_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_z: np.ndarray,
    xmin: float,
    ymin: float,
    cell_size: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Each point's elevation minus the ground estimate for its own cell, never negative."""
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)
    hag = z - ground_z[cell_id]
    return np.maximum(hag, 0.0)


def aggregate_voxels(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    hag: np.ndarray,
    intensity: np.ndarray,
    ground_z_per_point: np.ndarray,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    voxel_size: float,
) -> dict:
    """Group points into voxel_size cubes and summarize each one.

    Uses the file's own header minimum as the origin for every tile, not
    each tile's own local bounds, so voxels computed by different tiles line
    up on the exact same grid at shared edges instead of forming a seam.

    Every statistic here comes from a plain sum, count, sum, sum of squares,
    mean and standard deviation are derived from those afterward, on
    purpose: one pattern, reused for every column, nothing fancier needed.

    ground_z_per_point is each point's own cell's ground estimate, the same
    lookup height_above_ground already does, passed in rather than
    recomputed here, so every voxel also carries the ground level it was
    measured against, useful later for a downstream consumer without having
    to reconstruct it from height above ground and elevation.
    """
    vx = ((x - origin_x) / voxel_size).astype(np.int64)
    vy = ((y - origin_y) / voxel_size).astype(np.int64)
    vz = ((z - origin_z) / voxel_size).astype(np.int64)

    keys = np.stack([vx, vy, vz], axis=1)
    unique_keys, inverse, n_points = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n = len(unique_keys)

    def mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.bincount(inverse, weights=values, minlength=n)
        s2 = np.bincount(inverse, weights=values**2, minlength=n)
        mean = s / n_points
        variance = np.maximum(s2 / n_points - mean**2, 0.0)
        return mean, np.sqrt(variance)

    z_mean, z_std = mean_and_std(z)
    hag_mean, _ = mean_and_std(hag)
    intensity_mean, _ = mean_and_std(intensity)
    ground_z_mean, _ = mean_and_std(ground_z_per_point)

    return {
        "x_center": origin_x + (unique_keys[:, 0] + 0.5) * voxel_size,
        "y_center": origin_y + (unique_keys[:, 1] + 0.5) * voxel_size,
        "z_center": origin_z + (unique_keys[:, 2] + 0.5) * voxel_size,
        "n_points": n_points,
        "density": n_points / voxel_size**3,
        "z_mean": z_mean,
        "z_std": z_std,
        "hag_mean": hag_mean,
        "intensity_mean": intensity_mean,
        "ground_z": ground_z_mean,
    }


def drop_halo(voxels: dict, tile_xmin: float, tile_ymin: float, tile_xmax: float, tile_ymax: float) -> dict:
    """Keep only voxels whose centre falls inside the tile's own bounds, not its halo.

    This is what makes every voxel belong to exactly one tile: the halo
    exists purely to give the ground estimate real neighbours near the edge,
    once that is used its own voxels are thrown away here.
    """
    x_center = voxels["x_center"]
    y_center = voxels["y_center"]
    inside = (x_center >= tile_xmin) & (x_center < tile_xmax) & (y_center >= tile_ymin) & (y_center < tile_ymax)
    return {key: value[inside] for key, value in voxels.items()}
