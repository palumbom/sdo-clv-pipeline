"""Combine per-epoch tmp CSVs (from run_one.py / disBatch) into final CSVs.

run_one.py writes one timestamp-keyed file per epoch under <datadir>/tmp/;
this concatenates them (sorted by timestamp for deterministic order) into
<datadir>/{thresholds,region_output}.csv with headers.
"""

import argparse, os, glob
from sdo_clv_pipeline.sdo_io import (stitch_output_files, create_file,
                                     header_thresholds, header_region)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--keep-tmp", action="store_true",
                    help="do not delete the per-epoch tmp files after stitching")
    args = ap.parse_args()

    tmpdir = os.path.join(args.datadir, "tmp")
    thr_files = sorted(glob.glob(os.path.join(tmpdir, "thresholds_*.csv")))
    reg_files = sorted(glob.glob(os.path.join(tmpdir, "region_output_*.csv")))

    out_thr = os.path.join(args.datadir, "thresholds.csv")
    out_reg = os.path.join(args.datadir, "region_output.csv")
    create_file(out_thr, header_thresholds)
    create_file(out_reg, header_region)

    delete = not args.keep_tmp
    stitch_output_files(out_thr, thr_files, delete=delete)
    stitch_output_files(out_reg, reg_files, delete=delete)
    print(f"stitched {len(thr_files)} threshold rows and "
          f"{len(reg_files)} region files into {args.datadir}")
    return None


if __name__ == "__main__":
    main()
