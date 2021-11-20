# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 10:30h, 13/09/2018
#        Usage:
#
#               python Constrain.moveavg.py staid half-month
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
# -------------------------------------------------------------------------------
import sys

sys.path.append("../")
from src.Noise import Noise
from src import logger
from utils import find_time_duration

from os.path import join
import os
from functools import partial
import multiprocessing as mp
import sys

import matplotlib.pyplot as plt
from dateutil.rrule import rrule, MONTHLY
import numpy as np

plt.switch_backend("agg")


def constrain(
    staid, datadir="./measured", phase_band=(60, 120), min_ratio=3, half_month=3
):
    """constrain the seasonal ZH ratio with given selection rules

    Parameters
    ==========
    staid: str
        station ID
    datadir: str or path-like obj.
        data directory of all measured ZH ratio recording files
    phase_band: tuple
        phase band maintained for ZH ratio measurement
    min_ratio: float
        minimum HT ratio used for selection
    half_month: int
        half width in time for measuring ZH ratio
    """
    starttime, endtime = find_time_duration(staid, datadir)
    if starttime is np.nan:
        logger.error("No data error for {}".format(staid))
        return

    months = [dt for dt in rrule(MONTHLY, dtstart=starttime, until=endtime)]
    logger.info(
        "Meansuring ZH of {} during [{}-{}]".format(
            staid, months[0].strftime("%Y%m"), months[-1].strftime("%Y%m")
        )
    )
    # loop over each month
    for idx in range(half_month, len(months) - half_month):
        # Noise measurement class initialization
        nois = Noise(starttime=months[idx - 1], endtime=months[idx + 1], staid=staid)
        sub_months = months[idx - half_month : idx + half_month + 1]
        exdir = join(
            datadir,
            "Constrained",
            months[idx - half_month].strftime("%Y%m")
            + "_"
            + months[idx + half_month].strftime("%Y%m"),
        )
        os.makedirs(exdir, exist_ok=True)
        stalog = join(exdir, "{}.zh.csv".format(staid))
        if os.path.exists(stalog):
            logger.info(
                "file exist [ignore {}@{}]".format(
                    staid, sub_months[0].strftime("%Y%m")
                )
            )
            continue
        for sub_month in sub_months:
            monthid = sub_month.strftime("%Y%m")
            subdir = join(datadir, "{}".format(monthid))
            # search ZH logfile
            zhlogfile = join(subdir, "{}.zh.csv".format(staid))
            htlogfile = join(subdir, "{}.ht.csv".format(staid))
            phlogfile = join(subdir, "{}.dphi.csv".format(staid))
            azlogfile = join(subdir, "{}.az.csv".format(staid))
            # import measuremens
            nois.import_measurements(
                zhlogfile=zhlogfile,
                htlogfile=htlogfile,
                phlogfile=phlogfile,
                azlogfile=azlogfile,
            )
        # Check the imported result, if no result ignore below result
        if not nois.HT or not nois.ZH or not nois.DPHI:
            logger.info("NoData [ignore {}]".format(staid))
            continue
        # Constrain the results
        nois.criterions(
            phase_band=phase_band,
            min_ratio=min_ratio,
            bin=1,
            max_iter_nb=3,
            positive=False,
            maxzh=2.5,
            minzh=0.1,
        )
        nois.export(stalogfile=stalog, only_stalog=True)
        nois.plot_measurements(prefix=exdir)
        logger.info(
            "Suc. handle ZH of {} during [{}-{}]".format(
                staid,
                months[idx - half_month].strftime("%Y%m"),
                months[idx + half_month].strftime("%Y%m"),
            )
        )


if __name__ == "__main__":
    # Get stations
    with open(sys.argv[1]) as f:
        stas = [x.strip().split()[0] for x in f.readlines()]
    half_month = int(sys.argv[2])

    # Parallel for constraining the seasonal ZH ratios
    part_func = partial(constrain, datadir="./measured.new3", half_month=half_month)
    NB_PROCESSES = 45
    pool = mp.Pool(NB_PROCESSES)
    traces = pool.starmap(part_func, zip(stas))
    pool.close()
    pool.join()
