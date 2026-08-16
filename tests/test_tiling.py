"""Unit tests for the pure grid geometry in tiling.py.

build_tile_grid takes plain numbers, no COPC reader or network involved.
The octree estimation and zero-point dropping in plan_tiles is exercised
for real instead, against the actual file, since that logic only makes
sense with a real octree to query.
"""

from copc_pipeline.tiling import build_tile_grid


def _area(tile) -> float:
    return (tile.xmax - tile.xmin) * (tile.ymax - tile.ymin)


def _overlaps(a, b) -> bool:
    return not (a.xmax <= b.xmin or b.xmax <= a.xmin or a.ymax <= b.ymin or b.ymax <= a.ymin)


def test_grid_covers_extent_with_no_gaps_or_overlaps():
    tiles = build_tile_grid(xmin=0, ymin=0, xmax=8, ymax=8, n=4)

    assert len(tiles) == 16
    assert abs(sum(_area(t) for t in tiles) - 64.0) < 1e-9

    for i, a in enumerate(tiles):
        for b in tiles[i + 1 :]:
            assert not _overlaps(a, b), f"{a.tile_id} and {b.tile_id} overlap"


def test_grid_is_deterministic():
    first = build_tile_grid(xmin=0, ymin=0, xmax=8, ymax=8, n=4)
    second = build_tile_grid(xmin=0, ymin=0, xmax=8, ymax=8, n=4)

    assert first == second
    assert [t.tile_id for t in first] == [t.tile_id for t in second]
