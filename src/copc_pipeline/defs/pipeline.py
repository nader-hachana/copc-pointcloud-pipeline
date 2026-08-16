"""The real Dagster job: reads the source, plans tiles, fans out to process
each one, fans back in, and bulk loads the warehouse.

Every op reuses a function already built and verified in an earlier step,
this file's only job is wiring, opening a reader per op rather than passing
one between steps, since a live HTTP connection cannot be pickled between
processes the way the multiprocess executor needs to.
"""

import time

import numpy as np
from dagster import (
    AssetMaterialization,
    Backoff,
    DynamicOut,
    DynamicOutput,
    Failure,
    MetadataValue,
    OpExecutionContext,
    RetryPolicy,
    job,
    multiprocess_executor,
    op,
)
from laspy.copc import Bounds

from copc_pipeline.config import PipelineConfig
from copc_pipeline.enrich import (
    aggregate_voxels,
    compute_cell_id,
    compute_ground_grid,
    drop_halo,
    filter_noise,
    height_above_ground,
)
from copc_pipeline.metrics import peak_rss_mb
from copc_pipeline.source import SourceMetadata, open_reader, read_source_metadata
from copc_pipeline.storage import load_warehouse, write_tile_part
from copc_pipeline.tiling import TileSpec, plan_tiles


@op(retry_policy=RetryPolicy(max_retries=2, delay=5))
def read_source_metadata_op(context: OpExecutionContext) -> SourceMetadata:
    config = PipelineConfig()
    with open_reader(config.source_uri, http_num_threads=config.http_num_threads) as reader:
        metadata = read_source_metadata(reader)
    context.log.info(f"source has {metadata.point_count:,} points, copc_spacing {metadata.copc_spacing:.2f}")
    return metadata


@op(out=DynamicOut(TileSpec), retry_policy=RetryPolicy(max_retries=2, delay=5))
def plan_tiles_op(context: OpExecutionContext):
    config = PipelineConfig()
    with open_reader(config.source_uri, http_num_threads=config.http_num_threads) as reader:
        plan = plan_tiles(reader, config)

    if not plan.tiles:
        raise Failure(
            "Tile plan came back empty, nothing to process.",
            metadata={"source_uri": MetadataValue.text(config.source_uri)},
        )

    context.log.info(f"planned {len(plan.tiles)} tiles, overlap factor {plan.overlap_factor:.3f}")
    for tile in plan.tiles:
        yield DynamicOutput(tile, mapping_key=tile.tile_id)


@op(retry_policy=RetryPolicy(max_retries=3, delay=10, backoff=Backoff.EXPONENTIAL))
def process_tile_op(context: OpExecutionContext, tile: TileSpec) -> dict:
    """Stream one tile, enrich it, write its own Parquet part, return a small manifest row.

    Point arrays are born and die inside this one op, only the small
    manifest dict crosses back out, which is what keeps memory bounded
    under the multiprocess executor.
    """
    config = PipelineConfig()
    start = time.time()

    with open_reader(config.source_uri, http_num_threads=config.http_num_threads) as reader:
        h = reader.header
        halo = config.halo_m
        xmin, ymin, xmax, ymax = tile.xmin - halo, tile.ymin - halo, tile.xmax + halo, tile.ymax + halo

        fetch_bounds = Bounds(np.array([xmin, ymin]), np.array([xmax, ymax])).ensure_3d(h.mins, h.maxs)
        pts = reader.spatial_query(fetch_bounds)
        x, y, z = np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)
        intensity = np.asarray(pts.intensity).astype(np.float64)
        withheld = np.asarray(pts.withheld).astype(bool)
        points_fetched = len(x)

        keep, drop_counts = filter_noise(withheld)
        x, y, z, intensity = x[keep], y[keep], z[keep], intensity[keep]

        if len(x) == 0:
            n_voxels = 0
            points_in_voxels = 0
        else:
            ground_z, nx, ny = compute_ground_grid(x, y, z, xmin, ymin, xmax, ymax, config.ground_cell_size)
            hag = height_above_ground(x, y, z, ground_z, xmin, ymin, config.ground_cell_size, nx, ny)
            cell_id = compute_cell_id(x, y, xmin, ymin, config.ground_cell_size, nx, ny)
            ground_z_per_point = ground_z[cell_id]

            voxels = aggregate_voxels(
                x, y, z, hag, intensity, ground_z_per_point,
                float(h.mins[0]), float(h.mins[1]), float(h.mins[2]), config.voxel_size,
            )
            voxels = drop_halo(voxels, tile.xmin, tile.ymin, tile.xmax, tile.ymax)

            write_tile_part(voxels, tile.tile_id, config.parts_dir)
            n_voxels = len(voxels["n_points"])
            points_in_voxels = int(voxels["n_points"].sum())

    elapsed = time.time() - start
    manifest_row = {
        "tile_id": tile.tile_id,
        "points_fetched": points_fetched,
        "points_dropped_withheld": drop_counts["withheld"],
        "n_voxels": n_voxels,
        "points_in_voxels": points_in_voxels,
    }

    context.add_output_metadata(
        {
            "points_fetched": points_fetched,
            "n_voxels": n_voxels,
            "duration_s": round(elapsed, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
    )
    context.log.info(f"{tile.tile_id}: {n_voxels} voxels from {points_fetched:,} points in {elapsed:.2f}s")
    return manifest_row


@op
def collect_parts_op(context: OpExecutionContext, manifest_rows: list) -> list:
    total_voxels = sum(r["n_voxels"] for r in manifest_rows)
    context.log_event(
        AssetMaterialization(
            asset_key="tile_parts",
            metadata={
                "tiles_processed": len(manifest_rows),
                "total_voxels": total_voxels,
                "parts_dir": MetadataValue.path(str(PipelineConfig().parts_dir)),
            },
        )
    )
    context.log.info(f"collected {len(manifest_rows)} tile manifests, {total_voxels:,} voxels total")
    return manifest_rows


@op(retry_policy=RetryPolicy(max_retries=2, delay=5))
def load_warehouse_op(context: OpExecutionContext, manifest_rows: list, source_metadata: SourceMetadata) -> None:
    config = PipelineConfig()
    load_warehouse(config, manifest_rows, source_metadata)
    context.log_event(
        AssetMaterialization(
            asset_key="voxel_features_warehouse",
            metadata={
                "warehouse_path": MetadataValue.path(str(config.warehouse_path)),
                "tiles_loaded": len(manifest_rows),
            },
        )
    )
    context.log.info(f"warehouse loaded at {config.warehouse_path}")


@job(executor_def=multiprocess_executor.configured({"max_concurrent": PipelineConfig().max_concurrent}))
def copc_pipeline_job():
    source_metadata = read_source_metadata_op()
    tiles = plan_tiles_op()
    manifest_rows = tiles.map(process_tile_op).collect()
    collected = collect_parts_op(manifest_rows)
    load_warehouse_op(collected, source_metadata)
