"""Data quality checks against the loaded warehouse.

Each check is a plain SQL query against the real DuckDB warehouse, no new
concepts beyond what earlier steps already built. Meant to be run once after
load_warehouse, from the command line.
"""

import sys

import duckdb

from copc_pipeline.config import PipelineConfig


def run_checks(config: PipelineConfig) -> list[tuple[str, bool, str]]:
    con = duckdb.connect(str(config.warehouse_path))
    con.execute("LOAD spatial;")

    results = []

    n_voxels = con.execute("SELECT COUNT(*) FROM voxel_features").fetchone()[0]
    results.append(("warehouse has voxels", n_voxels > 0, f"{n_voxels:,} rows"))

    bad_rows = con.execute("""
        SELECT COUNT(*) FROM voxel_features
        WHERE n_points <= 0 OR x_center IS NULL OR y_center IS NULL OR z_center IS NULL
    """).fetchone()[0]
    results.append(("no empty or null voxels", bad_rows == 0, f"{bad_rows} bad rows"))

    negative_hag = con.execute("SELECT COUNT(*) FROM voxel_features WHERE hag_mean < 0").fetchone()[0]
    results.append(("height above ground never negative", negative_hag == 0, f"{negative_hag} negative rows"))

    mismatches = con.execute("""
        SELECT m.tile_id
        FROM tile_manifest m
        LEFT JOIN voxel_features v ON v.tile_id = m.tile_id
        GROUP BY m.tile_id, m.points_in_voxels
        HAVING m.points_in_voxels != COALESCE(SUM(v.n_points), 0)
    """).fetchall()
    results.append(("warehouse matches each tile's own manifest", len(mismatches) == 0, f"{len(mismatches)} tiles mismatched"))

    con.close()
    return results


def main() -> None:
    config = PipelineConfig()
    results = run_checks(config)

    all_passed = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
