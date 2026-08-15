"""Adapter around laspy's CopcReader for opening a COPC point cloud, local or remote.

CopcReader.open() already does the right thing for a remote source: an
http(s) uri gets wrapped in an HttpRangeStream, and point data is later
fetched by HttpFetcherThread workers, both issuing real Range: bytes=X-Y
requests rather than downloading the file as one block. This module only
wraps that call and adds a typed summary of the header and octree root
page, it does not change how any of the fetching itself works.
"""

from __future__ import annotations

from dataclasses import dataclass

from laspy.copc import CopcReader


def open_reader(uri: str, http_num_threads: int = 8) -> CopcReader:
    """Open a COPC source and read its header plus octree root page.

    Works as a context manager: `with open_reader(uri) as reader: ...`
    closes the underlying stream automatically.
    """
    return CopcReader.open(uri, http_num_threads=http_num_threads)


@dataclass(frozen=True)
class LevelStats:
    level: int
    node_count: int
    point_count: int


@dataclass(frozen=True)
class SourceMetadata:
    point_count: int
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    point_format_id: int
    point_size: int
    copc_spacing: float
    root_page_entries: int
    level_stats: list[LevelStats]


def read_source_metadata(reader: CopcReader) -> SourceMetadata:
    """Summarize a COPC source's header and octree root page.

    Only reads data already fetched when CopcReader.open() ran: the header
    and the root hierarchy page. No point data is touched here, so this is
    a handful of small http requests regardless of the source's total size.
    """
    header = reader.header

    level_counts: dict[int, list[int]] = {}
    for key, entry in reader.root_page.entries.items():
        counts = level_counts.setdefault(key.level, [0, 0])
        counts[0] += 1
        # some hierarchy entries carry a negative point_count as a sentinel
        # for "not loaded yet" rather than a real count, so only sum real ones
        if entry.point_count > 0:
            counts[1] += entry.point_count

    level_stats = [
        LevelStats(level=level, node_count=counts[0], point_count=counts[1])
        for level, counts in sorted(level_counts.items())
    ]

    return SourceMetadata(
        point_count=header.point_count,
        mins=tuple(float(v) for v in header.mins),
        maxs=tuple(float(v) for v in header.maxs),
        scales=tuple(float(v) for v in header.scales),
        offsets=tuple(float(v) for v in header.offsets),
        point_format_id=header.point_format.id,
        point_size=header.point_format.size,
        copc_spacing=float(reader.copc_info.spacing),
        root_page_entries=len(reader.root_page.entries),
        level_stats=level_stats,
    )
