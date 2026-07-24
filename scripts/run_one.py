"""Process a single epoch -- the robust building block for batch execution.

Unlike the in-process multiprocessing pool (run_pipe.py), this writes an
**idempotent** per-epoch output keyed by the observation timestamp (not the
worker PID), so:
  * reruns are safe and **resumable** (an already-done epoch is skipped), and
  * there is no stale-PID temp-file corruption.

It is meant to be driven one-epoch-per-task by disBatch (see
scripts/make_disbatch_tasks.py) for robust, resumable, failure-isolated batch
runs, or invoked directly for interactive single-epoch work. Internal numba
parallelism is controlled by the SDO_THREADS env var (default 1); for a single
interactive epoch set e.g. SDO_THREADS=16 for lower latency.

Outputs go to ``<datadir>/tmp/{thresholds,region_output,feature_output}_<stamp>.csv``
(stamp = YYYYmmddTHHMMSS); combine them afterwards with scripts/stitch_tmp.py.

Resume is gated on a per-epoch sentinel ``<datadir>/tmp/.done_<stamp>``, written only
after a successful, fully-written run -- not on the CSVs merely existing, since a
worker killed mid-write leaves them present but partial. The sentinel is a durable
done-log (the ``*.csv`` stitch globs ignore the dotfile); delete it to force an epoch
to reprocess.

Usage (explicit files):
  uv run scripts/run_one.py --con C.fits --mag M.fits --dop D.fits --aia A.fits --datadir DIR
Usage (index into a matched glob; used by the disBatch generator):
  uv run scripts/run_one.py --fitsdir DIR --globexp '2014*' --index 7 --datadir DIR
"""

import os, argparse, logging

from sdo_clv_pipeline.parallel import set_compute_threads
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import find_data, get_date
from sdo_clv_pipeline.sdo_process import process_data_set, status_ok
from sdo_clv_pipeline.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def resolve_epoch(args):
    """Return the (con, mag, dop, aia) paths for this task."""
    if args.index is not None:
        con, mag, dop, aia = find_data(args.fitsdir, globexp=args.globexp)
        n = len(con)
        assert 0 <= args.index < n, f"index {args.index} out of range [0, {n})"
        i = args.index
        return con[i], mag[i], dop[i], aia[i]
    assert args.con and args.mag and args.dop and args.aia, \
        "provide either --index (+--fitsdir/--globexp) or all of --con/--mag/--dop/--aia"
    return args.con, args.mag, args.dop, args.aia


def main():
    ap = argparse.ArgumentParser(description="process a single SDO epoch")
    ap.add_argument("--con"); ap.add_argument("--mag")
    ap.add_argument("--dop"); ap.add_argument("--aia")
    ap.add_argument("--fitsdir", default="/mnt/ceph/users/mpalumbo/sdo_data/")
    ap.add_argument("--globexp", default="")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--n-rings", type=int, default=10)
    ap.add_argument("--mu-thresh", type=float, default=0.1)
    ap.add_argument("--fit-cbs", action="store_true")
    ap.add_argument("--clobber", action="store_true",
                    help="reprocess even if this epoch's output already exists")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    configure_logging(args.log_level)
    set_compute_threads()  # SDO_THREADS env (default 1)

    con, mag, dop, aia = resolve_epoch(args)

    # idempotent per-epoch suffix from the observation timestamp
    stamp = get_date(con).strftime("%Y%m%dT%H%M%S")
    tmpdir = os.path.join(args.datadir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)  # process_data_set writes here but does not create it
    thr = os.path.join(tmpdir, f"thresholds_{stamp}.csv")
    reg = os.path.join(tmpdir, f"region_output_{stamp}.csv")
    feat = os.path.join(tmpdir, f"feature_output_{stamp}.csv")
    # Completion sentinel, written only after a successful process_data_set return.
    # The three CSVs existing is NOT proof of completion: process_data_set creates
    # them empty up front and fills them sequentially, so a worker killed mid-write
    # (OOM / SLURM preemption -- the failures this script exists to survive) leaves
    # all three present but partial. Gating resume on the sentinel instead means a
    # crashed epoch is reprocessed rather than silently skipped with truncated output.
    done = os.path.join(tmpdir, f".done_{stamp}")

    if not args.clobber and os.path.exists(done):
        logger.info("Epoch %s already done, skipping (resume)", stamp)
        print(f"skip {stamp}")
        return None

    # remove any partial/stale output for this epoch (and an orphaned sentinel) before
    # (re)writing -- covers a prior attempt killed before it wrote the sentinel
    for f in (thr, reg, feat, done):
        if os.path.exists(f):
            os.remove(f)

    status = process_data_set(con, mag, dop, aia, mu_thresh=args.mu_thresh,
                              n_rings=args.n_rings, suffix=stamp, datadir=args.datadir,
                              fit_cbs=args.fit_cbs, plot_moat=False, classify_moat=False)
    # mark done only on a fully-written success; a quality/error skip leaves no
    # sentinel and is re-attempted next run, exactly as before (skips are cheap and
    # may recover if a missing/corrupt input is later fixed)
    if status == status_ok:
        open(done, "w").close()
    print(f"{status} {stamp}")
    return None


if __name__ == "__main__":
    main()
