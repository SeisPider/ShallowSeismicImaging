import sys

sys.path.append("../")
from src.Noise import Noise
from src import logger

import os, sys
from obspy import read
from os.path import join, exists
from obspy import UTCDateTime
from dateutil.rrule import rrule, MONTHLY
import numpy as np

# Measure the ZH ratios from synthetic waveforms
def single_sta(staid, starttime, endtime, datadir=None):
    """Measure the ZH dispersion curve of specific station
    
    Parameters
    ==========
    staid: str
        station ID
    starttime: UTCDateTime
        start time for computation
    endtime: UTCDateTime
        end time for computation
    datadir: str or path-like obj.
        directory of instrumenal response removed continous
        waveform data
    """
    months = [dt for dt in rrule(MONTHLY, dtstart=starttime, until=endtime)]

    # Loop over month
    for month in months:
        # LOG info
        monthid = month.strftime("%Y%m")
        subdir = "./measured.new3/{}/".format(monthid)
        logger.info("Measuring {}@{}".format(staid, monthid))
        # check existence
        if exists(join(subdir, "{}.zh.csv".format(staid))):
            logger.info("File exists, ignore it !")
            continue

        try:
            st = read(join(datadir, "{}*".format(monthid), "{}*".format(staid)))
            if len(st) == 0:
                continue

            periods = np.concatenate((np.arange(3, 6, 0.2), np.arange(6.0, 10.0, 0.5)))
            freq_grid = 0.01
            nois = Noise(st)
            nois.CutSlices(timelen=3600, step_rate=0.05)
            nois.Measure_ratios(
                periods,             # Periods to compute ZH ratio
                freq_grid=freq_grid, # grid aournd the center frequency to compute ZH ratio
                nprocessor=45,       # Number of processors
                startup="fork",      # Method to start multiprocessing process, fork or spawn 
                                     # fork is more efficient and requirs more memory
                                     # spawn is to the opposite
                event_onset=9,       # Trigger onset in detecting earthquake-like events
                event_offset=7,     
            )

            # Export computed ZH ratio and associated parameters
            os.makedirs(subdir, exist_ok=True)
            nois.export(
                zhlogfile=join(subdir, "{}.zh.csv".format(staid)),
                htlogfile=join(subdir, "{}.ht.csv".format(staid)),
                phlogfile=join(subdir, "{}.dphi.csv".format(staid)),
                azlogfile=join(subdir, "{}.az.csv".format(staid)),
            )
        except Exception as err:
            logger.error("Unhandled Error [{}] for [{}]".format(err, staid))
        logger.info("Suc. Measured {}@{}".format(staid, monthid))


if __name__ == "__main__":
    # Get stations to compute
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    # obtain the time periods to compute ZH ratio
    stas = [line.strip().split()[0] for line in lines]
    startts = [UTCDateTime(line.strip().split()[-2]) for line in lines]
    endts = [UTCDateTime(line.strip().split()[-1]) for line in lines]

    datadir = "../data"  # directory of the daily waveforms

    for idx, sta in enumerate(stas):
        single_sta(sta, datadir=datadir, starttime=startts[idx], endtime=endts[idx])
