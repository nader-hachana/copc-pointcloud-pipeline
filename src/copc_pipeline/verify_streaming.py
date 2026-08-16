"""Streams the largest real tiles from the live S3 file and prints memory per tile.

This is the project's explicit checkpoint before building anything else on
top: prove memory stays flat while processing tiles, instead of climbing
tile after tile.

Deliberately tests the biggest tiles by estimate, not the first few in grid
order. On this file, testing plain grid order is misleading: one edge of the
16x16 grid ramps up steadily from an almost empty corner to a busy one, so
peak_rss_mb climbs simply because each tile is bigger than the last, nothing
to do with memory being retained. Testing the actual worst case tiles instead
shows the real signature of bounded memory: it rises briefly while the
process's working buffers warm up, then plateaus, even though every one of
these tiles holds a few million real points.
"""

import time

import numpy as np
from laspy.copc import Bounds

from copc_pipeline.config import PipelineConfig
from copc_pipeline.metrics import peak_rss_mb
from copc_pipeline.source import open_reader
from copc_pipeline.tiling import plan_tiles


def verify_streaming(config: PipelineConfig, sample_size: int = 10) -> None:
    with open_reader(config.source_uri, http_num_threads=config.http_num_threads) as reader:
        plan = plan_tiles(reader, config)
        sample = sorted(plan.tiles, key=lambda t: t.est_points, reverse=True)[:sample_size]

        print(f"{'tile':<8}{'points':>14}{'seconds':>10}{'peak_rss_mb':>14}")
        for tile in sample:
            start = time.time()

            bounds = Bounds(np.array([tile.xmin, tile.ymin]), np.array([tile.xmax, tile.ymax])).ensure_3d(
                reader.header.mins, reader.header.maxs
            )
            points = reader.spatial_query(bounds)

            elapsed = time.time() - start
            print(f"{tile.tile_id:<8}{len(points):>14,}{elapsed:>10.2f}{peak_rss_mb():>14.1f}")

            # dropping the reference here, before the next tile starts, is the
            # whole point: this tile's points must not still be held in memory
            # once the next one is fetched, or peak_rss_mb would climb instead
            # of staying flat
            del points


if __name__ == "__main__":
    verify_streaming(PipelineConfig())
