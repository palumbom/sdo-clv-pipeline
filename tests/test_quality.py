"""Tests for the QUALITY bitmask decoder."""

from sdo_clv_pipeline.quality import (decode_quality, format_quality, hmi_quality_bits,
                                      is_tolerable_quality, combine_quality,
                                      tolerable_quality_bits)


def test_clean_flag_returns_empty():
    assert decode_quality(0) == []
    assert decode_quality(0, "SDO/HMI") == []
    assert decode_quality(0, "SDO/AIA") == []


def test_hmi_decodes_known_bit_semantically():
    # bit 9 (0x200) == eclipse for HMI
    out = decode_quality(0x200, "SDO/HMI")
    assert len(out) == 1
    assert "bit 9" in out[0]
    assert "eclipse" in out[0]


def test_aia_reports_raw_bit_only():
    # AIA bit semantics are unverified: report position, no meaning text
    out = decode_quality(0x200, "SDO/AIA")
    assert out == ["bit 9"]


def test_multiple_bits_decoded():
    # bit 9 (eclipse) + bit 30 (target filtergram missing)
    q = (1 << 9) | (1 << 30)
    out = decode_quality(q, "SDO/HMI")
    assert len(out) == 2
    joined = "; ".join(out)
    assert "bit 9" in joined and "bit 30" in joined


def test_unknown_instrument_falls_back_to_raw_bits():
    out = decode_quality(0x200, "")
    assert out == ["bit 9"]


def test_format_quality_includes_hex_and_meaning():
    s = format_quality("SDO/HMI", 0x40000000)  # bit 30
    assert "0x40000000" in s
    assert "target filtergram missing" in s
    assert "SDO/HMI" in s


def test_format_quality_clean_flag():
    s = format_quality("SDO/HMI", 0)
    assert "0x00000000" in s
    assert "no bits set" in s


def test_hmi_table_only_has_documented_bits():
    # guard against accidental edits adding out-of-range bit positions
    assert all(0 <= b <= 31 for b in hmi_quality_bits)


def test_clean_quality_is_tolerable():
    assert is_tolerable_quality(0)


def test_soft_bits_are_tolerable():
    # each tolerable bit alone, and combined, must pass
    for b in tolerable_quality_bits:
        assert is_tolerable_quality(1 << b)
    combined = 0
    for b in tolerable_quality_bits:
        combined |= (1 << b)
    assert is_tolerable_quality(combined)


def test_fatal_bit_is_not_tolerable():
    assert not is_tolerable_quality(0x200)            # bit 9, eclipse


def test_one_fatal_bit_taints_otherwise_soft_flag():
    # bit 16 (tolerable) + bit 9 (fatal) -> not tolerable
    assert not is_tolerable_quality((1 << 16) | (1 << 9))


def test_combine_quality_is_bitwise_or():
    assert combine_quality(0x10000, 0, 0x10000, 0) == 0x10000
    assert combine_quality(0, 0, 0, 0) == 0
    assert combine_quality(1 << 16, 1 << 15, 0, 0) == (1 << 16) | (1 << 15)
