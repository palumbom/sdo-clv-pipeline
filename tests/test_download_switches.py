"""The fetch switches must be inspectable and short-circuit without a query."""

import inspect

import pytest

from sdo_clv_pipeline.sdo_download import download_data


def test_fetch_switches_exist_and_default_to_true():
    sig = inspect.signature(download_data)
    assert sig.parameters["fetch_hmi"].default is True
    assert sig.parameters["fetch_aia"].default is True


def test_both_switches_off_short_circuits_without_a_query():
    """Guards the caller contract: skipping both must not reach Fido.

    download_sdo_2024 relies on being able to call this unconditionally, so an
    all-skip call has to be free rather than a staged JSOC export of nothing.
    """
    result = download_data(series="720", email="nobody@example.com",
                           outdir="/nonexistent", start="2024-03-01T00:00:00",
                           end="2024-03-01T00:12:00", sample=0.2,
                           fetch_hmi=False, fetch_aia=False)
    assert result == ([], [], [], [])


def test_unknown_series_fails_loudly():
    """It used to return None, which callers discarded -- a typo'd series then
    silently no-opped every day of a run while each one counted as a success."""
    with pytest.raises(AssertionError, match="series"):
        download_data(series="bogus", email="nobody@example.com",
                      outdir="/nonexistent", start="2024-03-01T00:00:00",
                      end="2024-03-01T00:12:00", sample=0.2,
                      fetch_hmi=False, fetch_aia=False)
