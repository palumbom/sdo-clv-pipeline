"""Tests that organize_IO never truncates existing output on a non-clobber resume."""

import pytest

# organize_IO lives in sdo_io, which imports sunpy/astropy at module load.
pytest.importorskip("sunpy")
pytest.importorskip("astropy")

import csv
import os

from sdo_clv_pipeline import sdo_io
from sdo_clv_pipeline.sdo_io import (organize_IO, header_thresholds,
                                     header_region, header_feature)


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _row_count(path):
    with open(path) as f:
        return sum(1 for _ in f)


def test_resume_preserves_existing_output_when_feature_file_absent(tmp_path, monkeypatch):
    """A datadir from before feature_output.csv existed must keep its data rows.

    Reproduces the truncation bug: feature_output.csv was added to the resume
    gate, so a pre-existing (thresholds + region) datadir failed the all-exist
    check and fell to the branch that re-created *all* files in "w" mode,
    wiping thresholds.csv and region_output.csv.
    """
    # a real fits dir is required only to satisfy the isdir assert; the file
    # matching is stubbed so no FITS I/O happens.
    indir = tmp_path / "fits"
    indir.mkdir()
    monkeypatch.setattr(sdo_io, "find_data", lambda indir, globexp="": ([], [], [], []))

    datadir = tmp_path / "run"
    datadir.mkdir()
    thresh = datadir / "thresholds.csv"
    region = datadir / "region_output.csv"
    feature = datadir / "feature_output.csv"

    # seed a populated pre-feature-catalog datadir (no feature_output.csv)
    _write_csv(thresh, header_thresholds, [[56000.0] + [0.0] * (len(header_thresholds) - 1)])
    _write_csv(region, header_region, [[56000.0] + [1.0] * (len(header_region) - 1),
                                       [56000.0] + [2.0] * (len(header_region) - 1)])
    assert not feature.exists()

    organize_IO(str(indir), datadir=str(datadir), clobber=False, globexp="")

    # the pre-existing output must be untouched (header + its data rows)
    assert _row_count(thresh) == 2, "thresholds.csv lost its data row"
    assert _row_count(region) == 3, "region_output.csv lost its data rows"
    # the missing file is created, header-only
    assert feature.exists()
    assert _row_count(feature) == 1
    with open(feature) as f:
        assert next(csv.reader(f)) == header_feature


def test_resume_skips_already_processed_epochs(tmp_path, monkeypatch):
    """Epochs already recorded in thresholds.csv are removed from the work list."""
    from astropy.time import Time
    import datetime as dt

    indir = tmp_path / "fits"
    indir.mkdir()

    done = dt.datetime(2014, 1, 7, 12, 0, 0)
    todo = dt.datetime(2014, 1, 8, 12, 0, 0)

    def _names(d):
        s = d.strftime("%Y_%m_%d_%H_%M_%S")
        aia = "aia.lev1.1700." + d.strftime("%Y_%m_%dt%H_%M_%S") + ".fits"
        return (f"hmi.a.{s}.continuum.fits", f"hmi.a.{s}.magnetogram.fits",
                f"hmi.a.{s}.Dopplergram.fits", aia)

    con = [_names(done)[0], _names(todo)[0]]
    mag = [_names(done)[1], _names(todo)[1]]
    dop = [_names(done)[2], _names(todo)[2]]
    aia = [_names(done)[3], _names(todo)[3]]
    monkeypatch.setattr(sdo_io, "find_data",
                        lambda indir, globexp="": (con, mag, dop, aia))

    datadir = tmp_path / "run"
    datadir.mkdir()
    # thresholds.csv already contains the 'done' epoch
    mjd_done = Time(done).mjd
    _write_csv(datadir / "thresholds.csv", header_thresholds,
               [[mjd_done] + [0.0] * (len(header_thresholds) - 1)])
    _write_csv(datadir / "region_output.csv", header_region, [])
    _write_csv(datadir / "feature_output.csv", header_feature, [])

    con_out, mag_out, dop_out, aia_out = organize_IO(
        str(indir), datadir=str(datadir), clobber=False, globexp="")

    # the processed epoch is dropped from every instrument list; the new one stays
    assert con_out == [_names(todo)[0]]
    assert mag_out == [_names(todo)[1]]
    assert dop_out == [_names(todo)[2]]
    assert aia_out == [_names(todo)[3]]
