import os
import numpy as np
import argparse, warnings
from astropy.time import Time
from astropy import units as u
from sunpy.net import Fido, attrs as a
# from astropy.units.quantity import AstropyDeprecationWarning

email   = 'mlp95@psu.edu' # be kind

def download_data(series="720", email=None, outdir=None, start=None, end=None, sample=None, overwrite=False, progress=False):
    # setup arguments
    trange = a.Time(start, end)
    sample = a.Sample(sample * u.hour)
    provider = a.Provider("JSOC")
    notify = a.jsoc.Notify(email)
    # quality = a.jsoc.Keyword("QUALLEV1") == 0

    # set attributes for HMI query
    instr1 = a.Instrument.hmi
    if series == "45":
        physobs = (a.Physobs.intensity | a.Physobs.los_magnetic_field | a.Physobs.los_velocity)
    elif series == "720":
        physobs = (a.jsoc.Series("hmi.M_720s") | a.jsoc.Series("hmi.V_720s") | a.jsoc.Series("hmi.Ic_720s"))
    else:
        return None

    # get query for HMI and download data, retry failed downloads
    if series == "45":
        result = Fido.search(trange, instr1, physobs, sample)
    elif series == "720":
        result = Fido.search(trange, physobs, sample, notify)#, quality)
    else:
        return None

    print("About to fetch HMI files starting at date %s" % start)
    hmi_files = Fido.fetch(result, path=outdir, overwrite=overwrite, progress=progress)
    # while len(hmi_files.errors) > 0:
    #     hmi_files = Fido.fetch(hmi_files, path=outdir, overwrite=overwrite, progress=progress)

    # set attributes for AIA query
    level = a.Level(1)
    instr2 = a.Instrument.aia
    wavelength = a.Wavelength(1700. * u.AA)

    # get query for AIA and download data
    aia = Fido.search(trange, instr2, wavelength, level, provider, sample)
    print("About to fetch AIA files starting at date %s" % start)
    aia_files = Fido.fetch(aia, path=outdir, overwrite=overwrite, progress=progress)
    # while len(aia_files.errors) > 0:
    #     aia_files = Fido.fetch(aia_files, path=outdir, overwrite=overwrite, progress=progress)

    # sort out filenames into categories for output
    con_files = [s for s in hmi_files if "cont" in s]
    mag_files = [s for s in hmi_files if "magn" in s]
    dop_files = [s for s in hmi_files if "dopp" in s]
    aia_files = list(map(str, aia_files))
    return con_files, mag_files, dop_files, aia_files

def main():
    # supress warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # warnings.simplefilter(action='ignore', category=AstropyDeprecationWarning)

    # initialize argparser
    p = argparse.ArgumentParser(description='Download SDO/HMI data using sunpy')
    p.add_argument('--outdir', type=str, help='full directory path for file writeout')
    p.add_argument('-s', '--start_date', dest='start_date', type=str,   default='2024-01-01', help='Start date for query [yyyy-mm-dd]')
    p.add_argument('-e', '--stop_date',  dest='stop_date',  type=str,   default='2024-01-03', help='Stop date for query [yyyy-mm-dd]')
    p.add_argument('-c', '--cadence',    dest='cadence',    type=float, default=4, help='Cadence for query [hours]')
    p.add_argument('-o', '--overwrite',  dest='overwrite',  type=bool,  default=False, help='Whether to overwrite existing files')
    p.add_argument('-f', '--fast',       dest='fast',       type=bool,  default=False, help='Whether to use the fast (45 sec) data. Default is False (720 sec.)')
    args = p.parse_args()

    series = "45" if args.fast else "720"

    # Download per-day in order to catch failures 
    tstart = Time(args.start_date)
    tstop  = Time(args.stop_date)
    dates = np.arange(tstart, tstop+1*u.day, 1*u.day)
    outdir = os.path.abspath(args.outdir)

    # chunk the queries to manage lenght of file downloads
    for i in range(1, len(dates)):
        start = dates[i-1].isot
        end   = dates[i].isot
        
        try:
            files = download_data(series=series, outdir=outdir, email=email, 
                                  start=start, end=end, sample=args.cadence, 
                                  overwrite=args.overwrite)
        except Exception as e:
            print(f'Fido fetch failed for {start} to {end}')
            print(e)

    return None


if __name__ == "__main__":
    main()
