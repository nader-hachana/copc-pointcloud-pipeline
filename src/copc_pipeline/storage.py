"""Storage layer: per-tile Parquet parts, bulk loaded into a DuckDB warehouse.

Every tile's own voxel table gets written to its own Parquet file. One
single DuckDB statement then reads every part at once into a real table,
with a spatial index built on top. Nothing here ever inserts a row at a
time, tile parts, the manifest, and the source metadata table all load
through one bulk statement each.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from copc_pipeline.config import PipelineConfig
from copc_pipeline.source import SourceMetadata


def write_tile_part(voxels: dict, tile_id: str, parts_dir: Path) -> Path:
    """Write one tile's voxel table to its own Parquet part, zstd compressed.

    The path is deterministic from tile_id, so retrying a failed tile just
    overwrites its own file, safe to run more than once.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    path = parts_dir / f"tile_{tile_id}.parquet"
    n = len(voxels["n_points"])
    table = pa.table({**voxels, "tile_id": [tile_id] * n})
    pq.write_table(table, path, compression="zstd")
    return path


def load_warehouse(config: PipelineConfig, tile_manifest_rows: list[dict], source_metadata: SourceMetadata) -> None:
    """Bulk load every tile's Parquet part into one real DuckDB table.

    Also writes tile_manifest and source_metadata for provenance, and a
    cell_profile view collapsing voxels down to their 2D column, ground
    level, tallest point, tallest average height above ground, the natural
    thing a downstream consumer would actually want to query.
    """
    config.warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.warehouse_path))
    con.execute("INSTALL spatial; LOAD spatial;")

    parts_glob = str(config.parts_dir / "*.parquet")
    con.execute(f"""
        CREATE OR REPLACE TABLE voxel_features AS
        SELECT *, ST_Point(x_center, y_center) AS geom
        FROM read_parquet('{parts_glob}')
    """)
    con.execute("CREATE INDEX IF NOT EXISTS voxel_geom_idx ON voxel_features USING RTREE (geom)")

    manifest_table = pa.table(
        {
            "tile_id": [r["tile_id"] for r in tile_manifest_rows],
            "points_fetched": [r["points_fetched"] for r in tile_manifest_rows],
            "points_dropped_withheld": [r["points_dropped_withheld"] for r in tile_manifest_rows],
            "n_voxels": [r["n_voxels"] for r in tile_manifest_rows],
            "points_in_voxels": [r["points_in_voxels"] for r in tile_manifest_rows],
        }
    )
    con.execute("CREATE OR REPLACE TABLE tile_manifest AS SELECT * FROM manifest_table")

    source_table = pa.table(
        {
            "point_count": [source_metadata.point_count],
            "xmin": [source_metadata.mins[0]],
            "ymin": [source_metadata.mins[1]],
            "zmin": [source_metadata.mins[2]],
            "xmax": [source_metadata.maxs[0]],
            "ymax": [source_metadata.maxs[1]],
            "zmax": [source_metadata.maxs[2]],
            "copc_spacing": [source_metadata.copc_spacing],
        }
    )
    con.execute("CREATE OR REPLACE TABLE source_metadata AS SELECT * FROM source_table")

    con.execute("""
        CREATE OR REPLACE VIEW cell_profile AS
        SELECT
            x_center,
            y_center,
            MIN(ground_z) AS ground_z,
            MAX(z_center) AS max_z,
            MAX(hag_mean) AS max_height_above_ground,
            SUM(n_points) AS n_points
        FROM voxel_features
        GROUP BY x_center, y_center
    """)

    con.close()
