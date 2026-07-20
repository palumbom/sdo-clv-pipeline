"""Tests that the moat epoch cache is keyed correctly on mu_thresh."""

import pytest

# moat_cache pulls sdo_process -> sdo_image -> numba at import time.
pytest.importorskip("numba")
pytest.importorskip("astropy")

import numpy as np

from sdo_clv_pipeline import moat_cache
from sdo_clv_pipeline.moat_cache import cache_epoch, load_cached_epoch

# a filename get_date() can parse (HMI 45s format)
_CON = "hmi.a.2014_01_07_12_00_00.continuum.fits"
_MAG = "hmi.a.2014_01_07_12_00_00.magnetogram.fits"
_DOP = "hmi.a.2014_01_07_12_00_00.Dopplergram.fits"
_AIA = "aia.lev1.1700.2014_01_07t12_00_00.fits"


def test_cache_recomputes_when_mu_thresh_changes(tmp_path, monkeypatch):
    """A cache file computed at one mu_thresh must not be served for another.

    mu_thresh drives mask_low_mu and the cached invalid_mask/mu, so returning a
    file reduced at a different mu_thresh silently yields stale arrays.
    """
    calls = []

    def fake_reduce(con, mag, dop, aia, mu_thresh=0.1):
        calls.append(mu_thresh)
        return {"mu_thresh": float(mu_thresh), "dummy": np.zeros(2)}

    monkeypatch.setattr(moat_cache, "reduce_epoch_for_moats", fake_reduce)

    # first reduction at mu_thresh=0.1
    cache_epoch(_CON, _MAG, _DOP, _AIA, mu_thresh=0.1, cache_dir=tmp_path, clobber=False)
    assert calls == [0.1]

    # same epoch, different mu_thresh -> must recompute, not serve the 0.1 file
    p2 = cache_epoch(_CON, _MAG, _DOP, _AIA, mu_thresh=0.2, cache_dir=tmp_path, clobber=False)
    assert calls == [0.1, 0.2], "did not recompute for a new mu_thresh"
    assert load_cached_epoch(p2)["mu_thresh"] == pytest.approx(0.2)

    # same mu_thresh again -> genuine cache hit, no recompute
    cache_epoch(_CON, _MAG, _DOP, _AIA, mu_thresh=0.2, cache_dir=tmp_path, clobber=False)
    assert calls == [0.1, 0.2], "recomputed on a matching-mu_thresh hit"
