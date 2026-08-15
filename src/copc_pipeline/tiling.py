"""Octree-driven tile planner.

Divides a COPC source's XY extent into a tile_grid_n by tile_grid_n grid,
then adaptively splits any tile whose estimated point count exceeds
max_points_per_tile into 4 quadrants, recursively, up to a depth cap. Every
estimate comes from the octree hierarchy alone, load_octree_for_query never
fetches point data, so planning the whole file costs a handful of small
requests regardless of how many points it actually holds.

The grid and split logic (plan_tiles_grid) takes an estimate function as a
plain argument rather than reaching for a real CopcReader itself, so it can
be unit tested with a synthetic estimator and no network or file access at
all. plan_tiles is the real entry point that wires a live reader in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from laspy.copc import Bounds, CopcReader, load_octree_for_query

from copc_pipeline.config import PipelineConfig

# (xmin, ymin, xmax, ymax) -> (estimated_points, estimated_bytes, node_count)
EstimateFn = Callable[[float, float, float, float], tuple[int, int, int]]


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


def estimate_from_reader(reader: CopcReader) -> EstimateFn:
    """Build an estimate function backed by a real, already-open COPC reader.

    Each call queries the octree hierarchy for nodes overlapping the given
    XY box and sums their point_count and byte_size, without reading any
    point data. Nodes with point_count <= 0 are skipped, same convention as
    read_source_metadata's level_stats: a 0 means no points at that key, a
    negative value is a sentinel for "not loaded yet", neither is real data.
    """

    def _estimate(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[int, int, int]:
        bounds = Bounds(np.array([xmin, ymin]), np.array([xmax, ymax])).ensure_3d(
            reader.header.mins, reader.header.maxs
        )
        nodes = load_octree_for_query(
            reader.source,
            reader.copc_info,
            reader.root_page,
            query_bounds=bounds,
            level_range=None,
        )
        real_nodes = [n for n in nodes if n.point_count > 0]
        est_points = sum(n.point_count for n in real_nodes)
        est_bytes = sum(n.byte_size for n in real_nodes)
        return est_points, est_bytes, len(real_nodes)

    return _estimate


def _plan_cell(
    estimate_fn: EstimateFn,
    tile_id: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    depth: int,
    max_points_per_tile: int,
    max_split_depth: int,
) -> list[TileSpec]:
    est_points, est_bytes, node_count = estimate_fn(xmin, ymin, xmax, ymax)

    if est_points == 0:
        return []

    if est_points <= max_points_per_tile or depth >= max_split_depth:
        return [TileSpec(tile_id, xmin, ymin, xmax, ymax, est_points, est_bytes, node_count)]

    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    quadrants = [
        (f"{tile_id}_0", xmin, ymin, xmid, ymid),
        (f"{tile_id}_1", xmid, ymin, xmax, ymid),
        (f"{tile_id}_2", xmin, ymid, xmid, ymax),
        (f"{tile_id}_3", xmid, ymid, xmax, ymax),
    ]
    tiles: list[TileSpec] = []
    for qid, qxmin, qymin, qxmax, qymax in quadrants:
        tiles.extend(
            _plan_cell(
                estimate_fn, qid, qxmin, qymin, qxmax, qymax, depth + 1, max_points_per_tile, max_split_depth
            )
        )
    return tiles


def plan_tiles_grid(
    estimate_fn: EstimateFn,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    tile_grid_n: int,
    max_points_per_tile: int,
    max_split_depth: int,
) -> list[TileSpec]:
    """Pure grid and split logic, no COPC or network dependency.

    Builds a uniform tile_grid_n by tile_grid_n grid over the given extent,
    then hands each cell to the adaptive splitter. Tile ids are built purely
    from grid position and split path (t{i}_{j}, then _0/_1/_2/_3 per split),
    never from insertion order, so the same inputs always produce the same
    ids in the same order.
    """
    xs = np.linspace(xmin, xmax, tile_grid_n + 1)
    ys = np.linspace(ymin, ymax, tile_grid_n + 1)

    tiles: list[TileSpec] = []
    for i in range(tile_grid_n):
        for j in range(tile_grid_n):
            tiles.extend(
                _plan_cell(
                    estimate_fn,
                    f"t{i}_{j}",
                    float(xs[i]),
                    float(ys[j]),
                    float(xs[i + 1]),
                    float(ys[j + 1]),
                    0,
                    max_points_per_tile,
                    max_split_depth,
                )
            )
    return tiles


def plan_tiles(reader: CopcReader, config: PipelineConfig) -> TilePlan:
    """Plan tiles for a real, open COPC reader, using config's tunables."""
    header = reader.header
    tiles = plan_tiles_grid(
        estimate_from_reader(reader),
        float(header.mins[0]),
        float(header.mins[1]),
        float(header.maxs[0]),
        float(header.maxs[1]),
        config.tile_grid_n,
        config.max_points_per_tile,
        config.max_tile_split_depth,
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
