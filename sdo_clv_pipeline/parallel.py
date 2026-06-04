"""Runtime control of numba thread-parallelism for the compute kernels.

The heavy per-pixel kernels (geometry, bulk-velocity design matrix, bilinear
reproject) are compiled with ``parallel=True``. How many threads they use is a
*runtime* setting (``numba.set_num_threads``), so one build serves both modes:

  * **Across-epoch batch** (one epoch per process): 1 thread/process, so the
    process pool is not oversubscribed (N processes x M threads would thrash).
  * **Single-epoch / interactive** (e.g. the CBS sibling repo): many threads for
    low latency.

The package defaults to **1 thread** (set in ``__init__``), which is the safe
choice under the multiprocessing pool. Entry points opt into more via
``set_compute_threads`` (or the ``SDO_THREADS`` environment variable).
"""

import os
import numba


def set_compute_threads(n=None):
    """Set the numba thread count used by the parallel kernels.

    Parameters
    ----------
    n : int or None
        Thread count. If None, read the ``SDO_THREADS`` env var (default 1).
        Clamped to ``[1, NUMBA_NUM_THREADS]``. Returns the value actually set.
    """
    if n is None:
        n = int(os.environ.get("SDO_THREADS", "1"))
    n = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(n)
    return n


def get_compute_threads():
    """Return the current numba thread count."""
    return numba.get_num_threads()
