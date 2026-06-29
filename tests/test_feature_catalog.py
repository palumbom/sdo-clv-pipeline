"""Tests for the per-feature umbra/penumbra magnetic-field catalog."""

import pytest

# the catalog lives in sdo_vels, which imports sdo_image -> numba at import time;
# skip cleanly on the numba-free CI job rather than erroring at collection (G7).
pytest.importorskip("numba")

import numpy as np
from sdo_clv_pipeline.sdo_vels import compute_feature_catalog
from sdo_clv_pipeline.sdo_image import umbrae_code, penumbrae_code


# column positions in a catalog row (pre-quality_flag). Field columns carry both
# the line-of-sight (_los, Haywood proxy) and radial (_rad, B_obs/mu) conventions.
COL = {name: i for i, name in enumerate([
    "mjd", "region", "feature_id", "n_pix", "area_uhem", "mean_mu_iw",
    "min_mu", "max_mu", "centroid_lon", "centroid_lat",
    "mean_abs_b_iw_los", "mean_abs_b_aw_los", "max_abs_b_los",
    "mean_abs_b_iw_rad", "mean_abs_b_aw_rad", "max_abs_b_rad",
    "unsigned_flux_rad_g_uhem", "v_hat", "v_phot",
    "avg_int", "avg_int_flat"])}


def _toy_epoch():
    """A 3x5 frame: two separate umbra blobs and one penumbra blob.

    Layout (region codes; 0 = background):
        1 1 0 0 1
        0 0 0 0 1
        2 2 0 0 0
    Umbra blob A = {(0,0),(0,1)}, umbra blob B = {(0,4),(1,4)},
    penumbra blob = {(2,0),(2,1)}. With ndimage scan order, blob A -> label 1,
    blob B -> label 2. Intensity is set to 1.0 on all feature pixels so that the
    intensity-weighted means reduce to plain means and are easy to hand-check.
    The LOS and radial field arrays carry independent values so the _los/_rad
    columns are distinguishable.
    """
    u, p = umbrae_code, penumbrae_code
    regions = np.array([[u, u, 0, 0, u],
                        [0, 0, 0, 0, u],
                        [p, p, 0, 0, 0]], dtype=float)

    def grid(vals):
        # vals keyed by (row, col); background -> 0.0
        a = np.zeros((3, 5), dtype=float)
        for (r, c), v in vals.items():
            a[r, c] = v
        return a

    # blob A pixels (0,0),(0,1) carry the values we assert on
    intensity = grid({(0, 0): 1.0, (0, 1): 1.0, (0, 4): 1.0, (1, 4): 1.0,
                      (2, 0): 1.0, (2, 1): 1.0})
    abs_mag = grid({(0, 0): 100.0, (0, 1): 200.0, (0, 4): 50.0, (1, 4): 50.0,
                    (2, 0): 30.0, (2, 1): 30.0})
    abs_mag_rad = grid({(0, 0): 300.0, (0, 1): 400.0, (0, 4): 80.0, (1, 4): 80.0,
                        (2, 0): 60.0, (2, 1): 60.0})
    mu = grid({(0, 0): 0.8, (0, 1): 0.6, (0, 4): 0.9, (1, 4): 0.9,
               (2, 0): 0.5, (2, 1): 0.5})
    pix_area = grid({(0, 0): 2.0, (0, 1): 3.0, (0, 4): 1.0, (1, 4): 1.0,
                     (2, 0): 1.0, (2, 1): 1.0})
    lon = grid({(0, 0): 10.0, (0, 1): 20.0, (0, 4): 0.0, (1, 4): 0.0,
                (2, 0): 0.0, (2, 1): 0.0})
    lat = grid({(0, 0): 100.0, (0, 1): 110.0, (0, 4): 90.0, (1, 4): 90.0,
                (2, 0): 95.0, (2, 1): 95.0})
    iflat = grid({(0, 0): 0.5, (0, 1): 0.7, (0, 4): 0.4, (1, 4): 0.4,
                  (2, 0): 0.3, (2, 1): 0.3})
    p_vhat = grid({(0, 0): 4.0, (0, 1): 6.0, (0, 4): 2.0, (1, 4): 2.0,
                   (2, 0): 1.0, (2, 1): 1.0})
    p_vphot = grid({(0, 0): 1.0, (0, 1): 3.0, (0, 4): 0.5, (1, 4): 0.5,
                    (2, 0): 0.2, (2, 1): 0.2})

    return dict(mjd=12345.0, regions=regions,
                flat_int=intensity.ravel(), flat_iflat=iflat.ravel(),
                flat_mu=mu.ravel(), flat_abs_mag=abs_mag.ravel(),
                flat_abs_mag_rad=abs_mag_rad.ravel(),
                flat_pix_area=pix_area.ravel(), flat_lon=lon.ravel(),
                flat_lat=lat.ravel(), p_vhat=p_vhat.ravel(),
                p_vphot=p_vphot.ravel())


def test_labels_independent_umbra_and_penumbra_features():
    rows = compute_feature_catalog(**_toy_epoch())
    regions = [r[COL["region"]] for r in rows]
    assert regions.count(umbrae_code) == 2  # two separate umbra blobs
    assert regions.count(penumbrae_code) == 1  # one penumbra blob
    assert len(rows) == 3


def test_per_feature_statistics_match_hand_computed_values():
    rows = compute_feature_catalog(**_toy_epoch())
    # blob A: region=umbra, ndimage label 1
    a = next(r for r in rows
             if r[COL["region"]] == umbrae_code and r[COL["feature_id"]] == 1)

    assert a[COL["mjd"]] == 12345.0
    assert a[COL["n_pix"]] == 2
    assert a[COL["area_uhem"]] == pytest.approx(5.0)        # 2 + 3
    assert a[COL["mean_mu_iw"]] == pytest.approx(0.7)       # (0.8+0.6)/2
    assert a[COL["min_mu"]] == pytest.approx(0.6)
    assert a[COL["max_mu"]] == pytest.approx(0.8)
    assert a[COL["centroid_lon"]] == pytest.approx(15.0)    # (10+20)/2
    assert a[COL["centroid_lat"]] == pytest.approx(15.0)    # (100+110)/2 - 90 (Stonyhurst)

    # line-of-sight field (Haywood proxy)
    assert a[COL["mean_abs_b_iw_los"]] == pytest.approx(150.0)  # (100+200)/2
    assert a[COL["mean_abs_b_aw_los"]] == pytest.approx(160.0)  # (100*2+200*3)/5
    assert a[COL["max_abs_b_los"]] == pytest.approx(200.0)

    # radial field (B_obs/mu)
    assert a[COL["mean_abs_b_iw_rad"]] == pytest.approx(350.0)  # (300+400)/2
    assert a[COL["mean_abs_b_aw_rad"]] == pytest.approx(360.0)  # (300*2+400*3)/5
    assert a[COL["max_abs_b_rad"]] == pytest.approx(400.0)

    # radial unsigned flux = sum(|B_rad| * area)
    assert a[COL["unsigned_flux_rad_g_uhem"]] == pytest.approx(1800.0)  # 300*2+400*3

    assert a[COL["v_hat"]] == pytest.approx(5.0)            # (4+6)/2
    assert a[COL["v_phot"]] == pytest.approx(2.0)           # (1+3)/2
    assert a[COL["avg_int"]] == pytest.approx(1.0)
    assert a[COL["avg_int_flat"]] == pytest.approx(0.6)     # (0.5+0.7)/2


def test_empty_region_emits_no_rows():
    epoch = _toy_epoch()
    # wipe all penumbra pixels; umbra rows should remain, penumbra absent
    epoch["regions"][epoch["regions"] == penumbrae_code] = 0.0
    rows = compute_feature_catalog(**epoch)
    assert all(r[COL["region"]] != penumbrae_code for r in rows)
    assert any(r[COL["region"]] == umbrae_code for r in rows)
