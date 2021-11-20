# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 13:53h, 28/04/2018
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
from src import aicdpicker
from src.util import theoretical_arrival

from glob import glob
from os.path import join, basename
from obspy import UTCDateTime
import multiprocessing as mp
from itertools import repeat as rt
import subprocess as sp
import os

from obspy import read
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def determine_p(tr, min_freq, max_freq, plen=3, maxdiff=20, **kwargs):
    """determine arrival and waveform of P phase

    Parameter
    =========
    tr : obspy.trace
        trace to determine its P arrival and trim P phase
    """
    try:
        # tr.decimate(factor=10)
        tr.filter(type="bandpass", freqmin=min_freq, freqmax=max_freq)
        tr.detrend("linear")  # Perform a linear detrend on the data
        picker = aicdpicker.AICDPicker(**kwargs)

        # Pick P arrival
        scnl, picks, polarity, snr, uncert = picker.picks(tr)
        picks = np.array(picks)

        # get theoretical predicted arrivals
        origin, Parrobj = theoretical_arrival(tr)
        Parr = origin + Parrobj[0].time
        minidx = np.abs(picks - Parr).argmin()
        diff = picks[minidx] - Parr

        if np.abs(diff) <= maxdiff:
            return picks[minidx] - origin
        else:
            return None
    except Exception as err:
        print("Unhandled Error [{}]".format(err))
        return None


def P_arr_per_event(eventdir, starttime, endtime):
    """Estimate P phase arrivals of a prticular event

    Parameter
    =========
    eventdir : str
        directory of the event
    starttime : UTCDateTime
        time to start P arrival estimation
    endtime : UTCDateTime
        time to end P arrival estimation
    """
    eventid = basename(eventdir)
    print("Picking P arrivals for {}".format(eventid))
    origin = UTCDateTime(eventid)
    if origin <= starttime or origin >= endtime:
        print("Event exceeds insterested time interval skipping")
    try:
        st = read(join(eventdir, "*Z"))
    except FileNotFoundError:
        print("No data skipping")
        return
    except Exception as err:
        print("Unhandled Error [{}]".format(err))
        return

    arrs = []
    for tr in st:
        # determine the picking parameters
        p_phase = determine_p(
            tr,
            plen=3,
            min_freq=0.5,
            max_freq=2,
            t_ma=8,
            nsigma=8,
            t_up=0.78,
            nr_len=2,
            pol_len=5,
            pol_coeff=10,
            uncert_coeff=3,
        )
        arrs.append(p_phase)
        # write to SAC header user0
        if p_phase:
            os.putenv("SAC_DISPLAY_COPYRIGHT", "0")
            ZcompFile = join(eventdir, ".".join(tr.id.split(".")[0:2]) + "*Z")
            print(ZcompFile)
            sac = sp.Popen(["sac"], stdin=sp.PIPE)
            msg = "rh {}\n".format(ZcompFile)
            # erase previous result
            msg += "ch t5 {}\n".format(p_phase)
            msg += "wh\n"
            msg += "q\n"
            sac.communicate(msg.encode())


if __name__ == "__main__":
    NB_PROCESSES = 40
    starttime, endtime = UTCDateTime("19900101000000"), UTCDateTime("20180101000000")
    events = glob("../data/20*")

    # =========================================================================
    # Perform preprocess for each events
    # =========================================================================
    comblist = list(zip(events, rt(starttime), rt(endtime)))

    pool = mp.Pool(NB_PROCESSES)
    pool.starmap(P_arr_per_event, comblist)
    pool.close()
    pool.join()
