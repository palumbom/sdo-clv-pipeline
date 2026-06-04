"""Decode the SDO HMI/AIA QUALITY FITS keyword bitmask into human-readable causes."""

# HMI observables QUALITY bits, verified from JSOC documentation:
#   http://jsoc.stanford.edu/doc/data/hmi/Quality_Bits/QUALITY.txt
#   http://jsoc.stanford.edu/jsocwiki/Lev1qualBits
# AIA QUALITY bit semantics are NOT included here: the authoritative AIA02840
# keyword document was not accessible to verify them, so AIA flags are reported
# as raw bit positions only (see decode_quality).
hmi_quality_bits = {
    31: "no observables image produced (NODATA)",
    30: "target filtergram missing",
    29: "could not interpolate keywords (CROTA2/DSUN_OBS/CRLT_OBS)",
    28: "no framelist info / not an observables framelist",
    27: "wrong cadence for framelist",
    26: "wrong target (filtergram not in current framelist)",
    25: "not enough level-1d filtergrams",
    24: "missing keyword in level-1d (e.g. FID)",
    23: "wrong number of wavelengths in level-1d",
    22: "missing keyword in level-1p",
    21: "no look-up table record (MDI-like algorithm)",
    20: "could not read look-up table keywords",
    18: "not enough interpolation points (temporal interp)",
    17: "temporal interpolation failed",
    16: "low interpolation number (points > cadence apart)",
    15: "keywords could not be interpolated (closest-neighbor used)",
    14: "ISS loop OPEN for one or more filtergrams",
    13: "cannot read temperatures for polarization calibration",
    12: "could not gap-fill all level-1 filtergrams",
    11: "limb-fit issue (R_SUN/CRPIX missing or jittered)",
    10: "cosmic-ray hit list unreadable for some filtergrams",
     9: "eclipse (>=1 level-1 record during eclipse)",
     8: "large HFTSACID (>4000, adds noise)",
     7: "known code error (to be corrected in later versions)",
     5: "poor quality (eclipse/lunar transit/thermal recovery/open ISS/etc.)",
}


def decode_quality(quality, instrument=""):
    """Return a list of human-readable strings, one per set bit of `quality`.

    HMI instruments use the verified bit table; all others (notably AIA) report
    raw bit positions only, since their bit semantics are not authoritatively
    confirmed. A clean flag (quality == 0) returns an empty list.

    Parameters
    ----------
    quality : int
        The 32-bit QUALITY FITS keyword value.
    instrument : str, optional
        The TELESCOP keyword (e.g. "SDO/HMI", "SDO/AIA").
    """
    q = int(quality) & 0xFFFFFFFF
    set_bits = [i for i in range(32) if q & (1 << i)]
    is_hmi = "HMI" in (instrument or "").upper()
    out = []
    for b in set_bits:
        if is_hmi and b in hmi_quality_bits:
            out.append("bit %d: %s" % (b, hmi_quality_bits[b]))
        else:
            out.append("bit %d" % b)
    return out


def format_quality(instrument, quality):
    """Return a one-line summary like
    'SDO/HMI QUALITY=0x00000200 [bit 9: eclipse ...]'."""
    q = int(quality) & 0xFFFFFFFF
    causes = decode_quality(q, instrument)
    detail = "; ".join(causes) if causes else "no bits set"
    return "%s QUALITY=0x%08X [%s]" % (instrument, q, detail)
