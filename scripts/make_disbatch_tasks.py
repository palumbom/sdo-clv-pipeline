"""Generate a disBatch task file: one run_one.py invocation per matched epoch.

disBatch runs these tasks independently within a single SLURM allocation, with
failure isolation, retries, and resume (re-running the same task file skips
epochs whose output already exists, because run_one.py is idempotent). This is
the robust batch alternative to the in-process multiprocessing pool.

See https://github.com/flatironinstitute/disBatch and
https://wiki.flatironinstitute.org/SCC/Software/Slurm

Usage:
  uv run scripts/make_disbatch_tasks.py --fitsdir DIR --globexp '2014*' \
       --datadir data/2014 --threads 1 > tasks.db
Then, inside a SLURM allocation (e.g. `srun ... ` or an sbatch script):
  module load disBatch
  disBatch tasks.db          # add -r to retry failures
Afterwards combine the per-epoch outputs:
  uv run scripts/stitch_tmp.py --datadir data/2014
"""

import argparse
from sdo_clv_pipeline.paths import root
from sdo_clv_pipeline.sdo_io import find_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitsdir", default="/mnt/ceph/users/mpalumbo/sdo_data/")
    ap.add_argument("--globexp", default="")
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--threads", type=int, default=1,
                    help="numba threads per task (match cpus-per-task in the allocation)")
    ap.add_argument("--fit-cbs", action="store_true")
    args = ap.parse_args()

    con_files, _, _, _ = find_data(args.fitsdir, globexp=args.globexp)
    cbs = " --fit-cbs" if args.fit_cbs else ""
    for i in range(len(con_files)):
        print(f"cd {root} && SDO_THREADS={args.threads} uv run scripts/run_one.py "
              f"--fitsdir {args.fitsdir} --globexp '{args.globexp}' --index {i} "
              f"--datadir {args.datadir}{cbs}")
    return None


if __name__ == "__main__":
    main()
