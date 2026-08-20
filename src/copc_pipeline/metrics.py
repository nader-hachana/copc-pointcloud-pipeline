"""Peak memory reading used to prove streaming stays bounded.

Note: ru_maxrss is bytes on macOS/BSD, but kilobytes on Linux. Detect the
platform rather than assuming one, since this project should report correct
numbers wherever it actually runs.
"""

import resource
import sys


def peak_rss_mb() -> float:
    """Peak resident set size (real memory actually used) in megabytes, for this process so far."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1e3 if sys.platform.startswith("linux") else 1e6
    return raw / divisor
