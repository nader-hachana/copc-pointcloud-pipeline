"""Unit tests for the pure grid and split logic in tiling.py.

Every test here uses a synthetic estimate function, a plain Python callable
standing in for a real octree query, so none of this touches the network or
a real COPC file. plan_tiles itself, which wires a real CopcReader into
estimate_from_reader, is exercised for real in the Step 3 streaming
verification instead.
"""

from copc_pipeline.tiling import plan_tiles_grid


def _uniform_estimator(points_per_tile: int):
    def _estimate(xmin, ymin, xmax, ymax):
        return points_per_tile, points_per_tile * 30, 1

    return _estimate


def _area(tile) -> float:
    return (tile.xmax - tile.xmin) * (tile.ymax - tile.ymin)


def _overlaps(a, b) -> bool:
    return not (a.xmax <= b.xmin or b.xmax <= a.xmin or a.ymax <= b.ymin or b.ymax <= a.ymin)


def test_full_coverage_no_gaps_no_overlaps():
    tiles = plan_tiles_grid(
        _uniform_estimator(100),
        xmin=0,
        ymin=0,
        xmax=8,
        ymax=8,
        tile_grid_n=4,
        max_points_per_tile=1_000,
        max_split_depth=3,
    )

    assert len(tiles) == 16
    assert abs(sum(_area(t) for t in tiles) - 64.0) < 1e-9

    for i, a in enumerate(tiles):
        for b in tiles[i + 1 :]:
            assert not _overlaps(a, b), f"{a.tile_id} and {b.tile_id} overlap"


def test_split_triggers_only_where_needed():
    def estimate(xmin, ymin, xmax, ymax):
        # only the original bottom-left cell, before any splitting, is dense
        if xmin == 0 and ymin == 0 and xmax == 2 and ymax == 2:
            return 1_000, 1_000 * 30, 4
        return 10, 10 * 30, 1

    tiles = plan_tiles_grid(
        estimate, xmin=0, ymin=0, xmax=4, ymax=4, tile_grid_n=2, max_points_per_tile=100, max_split_depth=3
    )
    ids = {t.tile_id for t in tiles}

    assert "t0_0" not in ids, "the dense cell should have been split, not emitted whole"
    assert {"t0_0_0", "t0_0_1", "t0_0_2", "t0_0_3"} <= ids
    assert {"t0_1", "t1_0", "t1_1"} <= ids
    assert len(tiles) == 7

    # total area must still be exactly the original extent, split or not
    assert abs(sum(_area(t) for t in tiles) - 16.0) < 1e-9


def test_split_stops_at_depth_cap():
    tiles = plan_tiles_grid(
        _uniform_estimator(100_000),
        xmin=0,
        ymin=0,
        xmax=1,
        ymax=1,
        tile_grid_n=1,
        max_points_per_tile=100,
        max_split_depth=2,
    )

    # every tile is always over budget, so it must split exactly max_split_depth
    # times: 4 children, then 16 grandchildren, then stop
    assert len(tiles) == 16
    for t in tiles:
        assert t.tile_id.count("_") == 3, t.tile_id  # "t0_0" + 2 split segments


def test_zero_estimate_tiles_are_dropped():
    def estimate(xmin, ymin, xmax, ymax):
        if xmin == 0 and ymin == 0:
            return 0, 0, 0
        return 50, 50 * 30, 1

    tiles = plan_tiles_grid(
        estimate, xmin=0, ymin=0, xmax=4, ymax=4, tile_grid_n=2, max_points_per_tile=1_000, max_split_depth=3
    )
    ids = {t.tile_id for t in tiles}

    assert "t0_0" not in ids
    assert len(tiles) == 3


def test_ids_are_stable_and_deterministic():
    estimator = _uniform_estimator(100)
    kwargs = dict(xmin=0, ymin=0, xmax=8, ymax=8, tile_grid_n=4, max_points_per_tile=1_000, max_split_depth=3)

    first = plan_tiles_grid(estimator, **kwargs)
    second = plan_tiles_grid(estimator, **kwargs)

    assert [t.tile_id for t in first] == [t.tile_id for t in second]
    assert first == second
