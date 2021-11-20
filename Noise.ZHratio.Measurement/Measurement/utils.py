#!/usr/bin/env python
# Check directories to find the starttime and endtime
from os.path import join
from glob import glob

from obspy import UTCDateTime
import numpy as np

import sys
sys.path.append("../")
from src import logger

def find_time_duration(staid, datadir, pat="zh.csv"):
    """fint time duration of a specific seismic station

    Parameters
    ==========
    staid: str
        station ID
    datadir: str or path-like obj.
        data directory of all measured ZH ratio recording files
    """
    subdirs = glob(join(datadir, "[12]*", "{}.{}".format(staid, pat)))
    subdirs.sort()
    lenth = len(subdirs)
    if lenth == 0:
        fmt = "Duration time less than two months [{}]"
        logger.error(fmt.format(staid))
        return np.nan, np.nan
    elif lenth == 1:
        monthid = subdirs[0].split("/")[2]
        starttime = UTCDateTime(monthid + "01")
        return starttime, starttime 
    else:
        monthid1 = subdirs[0].split("/")[2]
        monthid2 = subdirs[-1].split("/")[2]
        starttime = UTCDateTime(monthid1 + "01")
        endtime = UTCDateTime(monthid2 + "01")
        return starttime, endtime

def import_zh(filename):
    """import ZH ration from zhratio log file

    Parameters
    ==========
    filename: str or path-like obj.
        filedirname of the log files
    """
    result = np.loadtxt(filename, skiprows=6)
    period, zh, std = result
    return period, zh, std

