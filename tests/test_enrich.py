"""Unit tests for enrich.py, one per function, small made up point clouds, no real file."""

import numpy as np

from copc_pipeline.enrich import (
    aggregate_voxels,
    compute_ground_grid,
    drop_halo,
    filter_noise,
    height_above_ground,
)


def test_filter_noise_drops_only_withheld_points():
    withheld = np.array([False, False, False, True, False])

    keep, counts = filter_noise(withheld)

    assert counts["withheld"] == 1
    assert list(keep) == [True, True, True, False, True]


def test_ground_grid_uses_a_low_percentile_and_fills_empty_cells():
    # cell (0,0): points at z=1..5, 10th percentile should sit near the low end
    # cell (1,0) has no points at all, it must get filled from its neighbour
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    y = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    ground_z, nx, ny = compute_ground_grid(x, y, z, xmin=0, ymin=0, xmax=2, ymax=1, cell_size=1, percentile=10.0)

    assert nx == 2 and ny == 1
    assert not np.isnan(ground_z).any(), "empty cell should have been filled, not left as NaN"
    assert ground_z[0] < 2.0  # a low percentile of 1..5 should sit near the bottom
    assert ground_z[1] == ground_z[0]  # filled from its only real neighbour


def test_height_above_ground_is_zero_or_positive():
    ground_z = np.array([10.0, 20.0])
    x = np.array([0.5, 1.5])
    y = np.array([0.5, 0.5])
    z = np.array([12.0, 19.0])  # second point is slightly below its own cell's ground estimate

    hag = height_above_ground(x, y, z, ground_z, xmin=0, ymin=0, cell_size=1, nx=2, ny=1)

    assert hag[0] == 2.0
    assert hag[1] == 0.0  # clamped, never negative


def test_aggregate_voxels_computes_correct_stats_for_one_voxel():
    # four points, all inside the same 1m voxel starting at the origin
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0.1, 0.2, 0.3, 0.4])
    z = np.array([0.1, 0.2, 0.3, 0.4])
    hag = np.array([1.0, 2.0, 3.0, 4.0])
    intensity = np.array([10.0, 20.0, 30.0, 40.0])

    voxels = aggregate_voxels(x, y, z, hag, intensity, origin_x=0, origin_y=0, origin_z=0, voxel_size=1.0)

    assert len(voxels["n_points"]) == 1
    assert voxels["n_points"][0] == 4
    assert voxels["z_mean"][0] == np.mean(z)
    assert voxels["hag_mean"][0] == np.mean(hag)
    assert voxels["x_center"][0] == 0.5  # centre of the voxel from 0 to 1


def test_drop_halo_keeps_only_voxels_inside_the_tiles_own_bounds():
    voxels = {
        "x_center": np.array([5.0, 15.0, 25.0]),
        "y_center": np.array([5.0, 5.0, 5.0]),
        "n_points": np.array([1, 1, 1]),
    }

    kept = drop_halo(voxels, tile_xmin=0, tile_ymin=0, tile_xmax=20, tile_ymax=10)

    assert list(kept["x_center"]) == [5.0, 15.0]
