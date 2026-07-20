"""Tests that clv_bfield consumes the pipeline's stored per-feature v_conv."""

import os
import sys

import pytest

# clv_bfield imports sdo_image (numba) and matplotlib at module load.
pytest.importorskip("numba")
pytest.importorskip("pandas")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sdo_clv_pipeline.paths import scripts
from sdo_clv_pipeline.sdo_image import umbrae_code

sys.path.insert(0, str(scripts))
import clv_bfield  # noqa: E402


def _feature_df(v_conv):
    return pd.DataFrame({
        "region": [umbrae_code] * len(v_conv),
        "mean_mu_iw": np.linspace(0.5, 0.9, len(v_conv)),
        "v_hat": np.arange(10.0, 10.0 + len(v_conv)),
        "v_quiet": np.zeros(len(v_conv)),
        "v_conv": np.asarray(v_conv, dtype=float),
        "n_pix": np.full(len(v_conv), 100),
    })


def test_prepare_features_uses_stored_vconv_and_drops_nothing():
    """The stored v_conv must survive unchanged and no feature is dropped."""
    stored = [1.5, -2.5, 3.5, -4.5]
    df = _feature_df(stored)
    out = clv_bfield.prepare_features(df)

    assert len(out) == len(stored), "features were dropped"
    assert list(out["v_conv"]) == stored, "stored v_conv was overwritten"


def test_prepare_features_requires_stored_vconv_column():
    """A feature_output.csv predating the stored v_conv column is a hard error."""
    df = _feature_df([1.0, 2.0]).drop(columns=["v_conv"])
    with pytest.raises(AssertionError):
        clv_bfield.prepare_features(df)
