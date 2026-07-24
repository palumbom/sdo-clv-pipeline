"""Real-data gates: full-frame oracle equivalence and the end-to-end goldens.

These are the checks that cannot run on synthetic frames -- they need a real WCS,
a real limb NaN pattern, a real region map, and the full 4096x4096 dynamic range.
They **skip automatically** when SDO FITS input is not reachable, so CI and any
machine without the data set stays green.

Point them elsewhere with `SDO_TEST_FITSDIR` and `SDO_TEST_GLOBEXP`.

Every `*_numpy` oracle retained in the package is compared here, so no oracle is
kept without something checking it. `scripts/benchmark_pipeline.py reproject`
covers the remaining one (`compute_pixel_mapping_highlevel`) as a stride sweep.

Regenerate the committed goldens after an intended numerical change with:

    SDO_REGEN_GOLDEN=1 uv run --extra test pytest tests/test_real_epoch.py -k golden

then inspect `git diff` on tests/fixtures/golden/ before keeping it.
"""

import glob, os, shutil

import pytest

pytest.importorskip("numba")

import numpy as np

from conftest import assert_within_gate

FITSDIR = os.environ.get("SDO_TEST_FITSDIR", "/mnt/ceph/users/mpalumbo/sdo_data/")
GLOBEXP = os.environ.get("SDO_TEST_GLOBEXP", "2014*01*07*")
REGEN = os.environ.get("SDO_REGEN_GOLDEN") == "1"

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "golden")
GOLDEN_NAMES = ("thresholds.csv", "region_output.csv", "feature_output.csv")
GOLDEN_EPOCHS = 3
MU_THRESH = 0.1
N_RINGS = 10


def _find_epochs(n):
    """Return n matched (con, mag, dop, aia) tuples, or skip if unavailable."""
    if not os.path.isdir(FITSDIR):
        pytest.skip("no SDO FITS data at %s (set SDO_TEST_FITSDIR)" % FITSDIR)
    from sdo_clv_pipeline.sdo_io import find_data
    con, mag, dop, aia = find_data(FITSDIR, globexp=GLOBEXP)
    if len(con) < n:
        pytest.skip("need %d epochs matching %r, found %d" % (n, GLOBEXP, len(con)))
    return list(zip(con, mag, dop, aia))[:n]


@pytest.fixture(scope="module")
def reduced_epoch():
    """One fully reduced epoch, shared by every oracle comparison in this module."""
    from sdo_clv_pipeline.sdo_process import reduce_sdo_images
    files = _find_epochs(1)[0]
    images = reduce_sdo_images(*files, mu_thresh=MU_THRESH)
    if images is None:
        pytest.skip("epoch did not reduce cleanly (quality skip?)")
    return files, images


# ---------------------------------------------------------------------------
# oracle equivalence on a real full frame
# ---------------------------------------------------------------------------

def test_pixel_area_matches_oracle_on_real_geometry(reduced_epoch):
    from sdo_clv_pipeline.sdo_image import calculate_pixel_area_numpy
    _files, (con, mag, dop, aia, mask) = reduced_epoch
    assert_within_gate(dop.pix_area,
                       calculate_pixel_area_numpy(dop.lat, dop.lon),
                       "dop.pix_area")
    return None


def test_spacecraft_vel_matches_oracle_on_real_geometry(reduced_epoch):
    """Both paths are recomputed from rr/xx/yy/mask_nan, none of which
    mask_low_mu touches, so this compares implementations rather than states."""
    _files, (con, mag, dop, aia, mask) = reduced_epoch
    dop.calc_spacecraft_vel()
    fast = dop.v_obs.copy()
    dop.calc_spacecraft_vel_numpy()
    oracle = dop.v_obs
    dop.v_obs = fast                      # restore for any later comparison
    assert_within_gate(fast, oracle, "dop.v_obs")
    return None


def test_limb_darkening_matches_oracle_on_real_frames(reduced_epoch):
    """Continuum is reloaded so the fit sees the unmasked frame production sees;
    the AIA image is reused post-reduction to avoid a second ~3 s reprojection."""
    from sdo_clv_pipeline.sdo_image import SDOImage
    _files, (con, mag, dop, aia, mask) = reduced_epoch

    fresh = SDOImage(con.filename)
    fresh.inherit_geometry(dop)
    for label, img in (("con", fresh), ("aia", aia)):
        img.calc_limb_darkening()
        fast = (img.ld_coeffs.copy(), img.ldark.copy(), img.iflat.copy())
        img.calc_limb_darkening_numpy()
        assert_within_gate(fast[0], img.ld_coeffs, "%s.ld_coeffs" % label)
        assert_within_gate(fast[1], img.ldark, "%s.ldark" % label)
        assert_within_gate(fast[2], img.iflat, "%s.iflat" % label)
    return None


def test_geometry_matches_numpy_oracle_on_real_wcs(reduced_epoch):
    """The fused analytic kernel vs the vectorized numpy chain, on a real WCS.

    Previously unchecked by any test: the synthetic-frame suite cannot exercise a
    real HPLN/HPLT-TAN WCS or the off-disk NaN boundary.
    """
    from sdo_clv_pipeline.sdo_image import SDOImage
    _files, (con, mag, dop, aia, mask) = reduced_epoch

    img = SDOImage(dop.filename)
    img.calc_geometry()
    fast = {k: np.copy(getattr(img, k)) for k in ("mu", "pix_area")}
    fast["lat"] = img.lat.value.copy()
    fast["lon"] = img.lon.value.copy()
    fast["rr"] = img.rr.value.copy()

    img.calc_geometry_numpy()
    # These are two different algorithms, not a fused rewrite of one: the numba
    # path does an inline rotated-TAN inverse, the numpy path calls astropy's
    # wcs_pix2world. They agree to ~5e-10 relative in rr (float64); mu is stored
    # as float32, so that lands within a couple of ULP and needs the ULP floor
    # rather than the default 1e-12, which is tighter than float32 can represent.
    # Measured agreement, as a fraction of each array's own scale: rr 4.5e-10,
    # lat 1.5e-10, lon 1.6e-9, pix_area 6.5e-9. In absolute terms the worst
    # longitude pixel differs by 2.8e-7 deg -- 1 milliarcsecond, about 1/500 of an
    # HMI pixel -- and pix_area amplifies that because it is built from first
    # differences of lat/lon. The tolerances below sit just above the measured
    # values so a real change trips them.
    assert_within_gate(fast["mu"], img.mu, "geometry mu", ulp_floor=True)
    assert_within_gate(fast["rr"], img.rr.value, "geometry rr", rtol=1e-9)
    assert_within_gate(fast["lat"], img.lat.value, "geometry lat", rtol=1e-9)
    assert_within_gate(fast["lon"], img.lon.value, "geometry lon", rtol=1e-8)
    assert_within_gate(fast["pix_area"], img.pix_area, "geometry pix_area",
                       rtol=1e-8)
    return None


def test_bulk_vel_matches_numpy_oracle_on_real_dopplergram(reduced_epoch):
    """The fused Legendre design-matrix path vs the scipy/gen_leg oracle.

    ``calc_bulk_vel_numpy`` had no caller before this test, so the ~90 lines of
    reference implementation it holds were unverified. Runs both cbs settings
    because they build differently sized systems (6 vs 11 terms).
    """
    from sdo_clv_pipeline.sdo_image import SDOImage
    _files, (con, mag, dop, aia, mask) = reduced_epoch

    for fit_cbs in (False, True):
        img = SDOImage(dop.filename)
        img.calc_geometry()
        img.mask_nan = (img.mu >= MU_THRESH)
        img.v_grav = 633
        img.calc_spacecraft_vel()

        img.calc_bulk_vel(fit_cbs=fit_cbs)
        fast = {k: np.copy(getattr(img, k))
                for k in ("v_rot", "v_mer", "v_cbs", "v_corr")}
        fast["fit_params"] = img.fit_params.copy()

        img.calc_bulk_vel_numpy(fit_cbs=fit_cbs)
        tag = "fit_cbs=%s" % fit_cbs
        # The design matrices come from different routines (an in-kernel Legendre
        # recurrence vs scipy.special.eval_legendre), so this is an
        # algorithm-vs-algorithm comparison. fit_params agrees to ~7e-14 for the
        # 6-term system and ~3e-10 for the 11-term CBS one, whose normal-equations
        # matrix has cond ~4e7 -- hence the 1e-9 allowance. The reconstructed
        # velocity fields are float32, so they get the ULP floor.
        assert_within_gate(fast["fit_params"], img.fit_params,
                           "bulk fit_params (%s)" % tag, rtol=1e-9)
        for k in ("v_rot", "v_mer", "v_cbs", "v_corr"):
            assert_within_gate(fast[k], getattr(img, k), "%s (%s)" % (k, tag),
                               ulp_floor=True)
    return None


# ---------------------------------------------------------------------------
# end-to-end goldens
# ---------------------------------------------------------------------------

def _produce_csvs(epochs, outdir):
    """Run each epoch through process_data_set, then stitch the per-epoch parts."""
    from sdo_clv_pipeline.sdo_io import stitch_output_files, create_file
    from sdo_clv_pipeline.sdo_process import process_data_set

    os.makedirs(os.path.join(outdir, "tmp"), exist_ok=True)
    for i, files in enumerate(epochs):
        status = process_data_set(*files, mu_thresh=MU_THRESH, n_rings=N_RINGS,
                                 suffix="g%02d" % i, datadir=outdir,
                                 plot_moat=False, classify_moat=False)
        assert status == "ok", "epoch %d status=%r" % (i, status)

    for name in GOLDEN_NAMES:
        target = os.path.join(outdir, name)
        create_file(target)
        stem = name[:-len(".csv")]
        parts = sorted(glob.glob(os.path.join(outdir, "tmp", stem + "_g*.csv")))
        stitch_output_files(target, parts, delete=False)
    return None


def test_end_to_end_output_matches_goldens(tmp_path):
    """The only gate covering the whole chain, including the CSV writer.

    Compared column-by-column under the project tolerance. Most columns are
    bit-identical; the region-only rows sum mu-ring partials and so land a few
    times 1e-14 away, which is inside the gate.
    """
    pd = pytest.importorskip("pandas")
    epochs = _find_epochs(GOLDEN_EPOCHS)
    workdir = str(tmp_path / "golden")
    os.makedirs(workdir)
    _produce_csvs(epochs, workdir)

    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        for name in GOLDEN_NAMES:
            shutil.copy(os.path.join(workdir, name), os.path.join(GOLDEN_DIR, name))
        pytest.skip("SDO_REGEN_GOLDEN=1: goldens rewritten, review git diff")

    missing = [n for n in GOLDEN_NAMES
               if not os.path.exists(os.path.join(GOLDEN_DIR, n))]
    if missing:
        pytest.skip("no goldens for %s (run with SDO_REGEN_GOLDEN=1)" % ", ".join(missing))

    for name in GOLDEN_NAMES:
        new = pd.read_csv(os.path.join(workdir, name), header=None)
        old = pd.read_csv(os.path.join(GOLDEN_DIR, name), header=None)
        assert new.shape == old.shape, \
            "%s: shape %s != golden %s (row or column count changed)" % (
                name, new.shape, old.shape)
        for col in range(new.shape[1]):
            assert_within_gate(new.iloc[:, col].to_numpy(dtype=float),
                               old.iloc[:, col].to_numpy(dtype=float),
                               "%s col %d" % (name, col))
    return None
