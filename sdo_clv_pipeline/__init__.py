# Default the parallel numba kernels to a single thread. This is the safe
# setting under the across-epoch multiprocessing pool (one epoch per process);
# single-epoch / interactive entry points opt into more threads via
# parallel.set_compute_threads() or the SDO_THREADS env var.
#
# Guarded so pure-Python modules (e.g. quality) stay importable in lightweight
# environments without numba installed, such as the --no-deps CI test job.
try:
    from .parallel import set_compute_threads
    set_compute_threads(1)
except ImportError:
    pass
