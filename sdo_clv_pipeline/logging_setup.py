"""Shared logging configuration for the SDO CLV pipeline and its consumers."""

import logging

# standard timestamped format; SLURM captures the stderr stream this writes to
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"


def configure_logging(level=logging.INFO):
    """Configure root logging to stderr with a timestamped format.

    Idempotent: existing handlers are removed before adding ours, so repeated
    calls (or use as a multiprocessing Pool initializer, which runs once per
    spawned worker) do not duplicate log lines.

    Parameters
    ----------
    level : int or str, optional
        Logging level (e.g. logging.INFO or "INFO"). Strings are accepted so a
        CLI --log-level value can be passed straight through.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root = logging.getLogger()
    root.setLevel(level)

    # drop any pre-existing handlers so re-config / per-worker init stays clean
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler()  # stderr
    handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root.addHandler(handler)

    # route un-silenced astropy/sunpy warnings through logging, and quiet noisy libs
    logging.captureWarnings(True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return None
