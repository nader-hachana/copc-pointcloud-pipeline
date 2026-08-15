from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    """Every tunable parameter the pipeline needs, in one place.

    Any field can be overridden through an environment variable prefixed
    with COPC_, for example COPC_TILE_GRID_N=8, without touching code.
    """

    model_config = SettingsConfigDict(env_prefix="COPC_", frozen=True)

    source_uri: str = "https://s3.amazonaws.com/hobu-lidar/sofi.copc.laz"

    tile_grid_n: int = 16
    max_points_per_tile: int = 6_000_000
    max_tile_split_depth: int = 3

    voxel_size: float = 1.0
    ground_cell_size: float = 2.0
    halo_m: float = 5.0

    resolution: Optional[float] = None

    http_num_threads: int = 8
    max_concurrent: int = 4
    tile_limit: Optional[int] = None

    data_dir: Path = Path("data")
    warehouse_path: Path = Path("warehouse/copc_pipeline.duckdb")

    @property
    def parts_dir(self) -> Path:
        return self.data_dir / "parts"
