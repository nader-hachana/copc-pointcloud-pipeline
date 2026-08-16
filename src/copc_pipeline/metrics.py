"""Peak memory reading used to prove streaming stays bounded.

Note: ru_maxrss is bytes on macOS/BSD, but kilobytes on Linux. This project
runs on macOS, where the raw value divided by 1e6 gives megabytes directly.
Running this on Linux would need dividing by 1e3 instead.
"""

import resource


def peak_rss_mb() -> float:
    """Peak resident set size (real memory actually used) in megabytes, for this process so far."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
