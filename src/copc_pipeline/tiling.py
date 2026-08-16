"""Tile planner.

Divides a COPC source's XY extent into a uniform tile_grid_n by tile_grid_n
grid, then asks the octree hierarchy how many points and bytes fall in each
tile, without fetching any point data. This is what keeps the pipeline's
memory bounded: each tile gets streamed and processed on its own later,
instead of loading the whole file at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from laspy.copc import Bounds, CopcReader, load_octree_for_query

from copc_pipeline.config import PipelineConfig


@dataclass(frozen=True)
class TileBounds:
    tile_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    est_points: int
    est_bytes: int
    node_count: int


@dataclass(frozen=True)
class TilePlan:
    tiles: list[TileSpec]
    source_point_count: int
    total_estimated_points: int
    overlap_factor: float


def build_tile_grid(xmin: float, ymin: float, xmax: float, ymax: float, n: int) -> list[TileBounds]:
    """Split a bounding box into an n by n grid.

    Pure geometry, no COPC or network involved, so this is directly unit
    testable with plain numbers.
    """
    xs = np.linspace(xmin, xmax, n + 1)
    ys = np.linspace(ymin, ymax, n + 1)

    tiles: list[TileBounds] = []
    for i in range(n):
        for j in range(n):
            tiles.append(
                TileBounds(f"t{i}_{j}", float(xs[i]), float(ys[j]), float(xs[i + 1]), float(ys[j + 1]))
            )
    return tiles


def _estimate_tile(reader: CopcReader, tile: TileBounds) -> tuple[int, int, int]:
    """Ask the octree hierarchy for the nodes overlapping this tile.

    Returns (estimated_points, estimated_bytes, node_count). Nodes with
    point_count <= 0 are skipped: 0 means no points at that key, -1 is a
    sentinel meaning the real entry lives in a child page, neither is real
    point data.
    """
    bounds = Bounds(np.array([tile.xmin, tile.ymin]), np.array([tile.xmax, tile.ymax])).ensure_3d(
        reader.header.mins, reader.header.maxs
    )
    nodes = load_octree_for_query(
        reader.source, reader.copc_info, reader.root_page, query_bounds=bounds, level_range=None
    )
    real_nodes = [node for node in nodes if node.point_count > 0]
    return sum(node.point_count for node in real_nodes), sum(node.byte_size for node in real_nodes), len(real_nodes)


def plan_tiles(reader: CopcReader, config: PipelineConfig) -> TilePlan:
    """Plan tiles for a real, open COPC reader.

    Every estimate comes from the octree hierarchy already loaded when the
    reader was opened, so planning costs no extra point-data fetches.
    Tiles with zero estimated points are dropped.
    """
    header = reader.header
    grid = build_tile_grid(
        float(header.mins[0]), float(header.mins[1]), float(header.maxs[0]), float(header.maxs[1]), config.tile_grid_n
    )

    tiles: list[TileSpec] = []
    for tile in grid:
        est_points, est_bytes, node_count = _estimate_tile(reader, tile)
        if est_points == 0:
            continue
        tiles.append(
            TileSpec(tile.tile_id, tile.xmin, tile.ymin, tile.xmax, tile.ymax, est_points, est_bytes, node_count)
        )

    if config.tile_limit is not None:
        tiles = tiles[: config.tile_limit]

    total_estimated = sum(t.est_points for t in tiles)
    overlap_factor = total_estimated / header.point_count if header.point_count else 0.0

    return TilePlan(
        tiles=tiles,
        source_point_count=header.point_count,
        total_estimated_points=total_estimated,
        overlap_factor=overlap_factor,
    )
