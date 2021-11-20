# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 19:26h, 12/11/2018
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
# -------------------------------------------------------------------------------
"""Decompose the total ZH ratio sery into four parts: time-independent average
, seasonal variation, linear variation and gaussian residual part
"""
import sys

sys.path.append("../")
from src import logger
from utils import find_time_duration, import_zh

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
from dateutil.rrule import rrule, MONTHLY
from scipy.optimize import curve_fit

import sys
from os.path import join, exists
import os
from functools import partial
import multiprocessing as mp

plt.switch_backend("agg")


def curve_fitness(data, std, period):
    """Fit the sine curve and output residual

    Parameters
    ==========
    data: numpy.array
        Time sery of time-independent ZH ratio
    std: numpy.array
        Measurement uncertainty of each seasonal ZH ratio
    period: float
        period of these ZH ratio
    """
    # time scale
    twopi = 2 * np.pi
    t = np.arange(1, len(data) + 1) * twopi

    # Delete unreliable measurements
    msk = ~np.isnan(data)
    t, data, std = t[msk], data[msk], std[msk]

    # Delete no data samples
    msk = data != 0
    t, data, std = t[msk], data[msk], std[msk]

    # Delete outlier
    msk = np.abs(data - np.mean(data)) > 3 * np.std(data)
    data[msk] = 0

    # Delete no data samples
    msk = data != 0
    t, data, std = t[msk], data[msk], std[msk]

    # Guess the initial solution
    guess_mean, guess_amp = np.mean(data), np.std(data)
    guess_phase = 0

    # Curve fitting to investigate parameters
    def func(x, a, b, c):
        return a * np.sin(x / 12 + b) + c

    result = curve_fit(
        func,
        t,
        data,
        sigma=std,
        full_output=1,
        p0=[guess_amp, guess_phase, guess_mean],
        absolute_sigma=True,
        maxfev=200000,
    )
    popt, pcov = result[0], result[1]
    res = data - func(t, *popt)
    return popt, np.sqrt(np.diag(pcov)), res.std()


def visulize_decomposed_result(staid, datadir, months, rawzhs, pers, periods):
    """Same as the name

    Parameters
    ==========
    staid: str
        station ID
    datadir: str or path-like obj.
        directory of the constrained ZH ratios
    months: list
        all months
    rawzhs: numpy.array
        raw mean of seasonal zh ratios
    pers: numpy.array
        periods to show detail, (raw, decomposed curve) and so on
    periods: numpy.array
        periods of the whole dispersion curve
    """
    N = len(pers)
    cmap = plt.cm.coolwarm
    rcParams["axes.prop_cycle"] = cycler(color=cmap(np.linspace(0, 1, N)))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))
    for per in pers:
        idx = np.abs(per - periods).argmin()
        per_zh = rawzhs[:, idx]

        # ignore invalid measurements
        msk = per_zh != 0
        axes[0].plot(
            np.array(months)[msk],
            per_zh[msk],
            linestyle="--",
            marker="o",
            label="{}s".format(per),
        )
    axes[0].set_xlabel("Time (Year-month)")
    axes[0].set_ylabel("ZH ratio")
    axes[0].legend()
    axes[0].set_title("ZH ratio time variation @{}".format(staid))

    # import fitted cosine func. and ZH ratio curve
    filename = join(datadir, "Decomposed", "{}.csv".format(staid))
    result = np.loadtxt(filename, unpack=True, skiprows=1, usecols=(0, 1, 3, 4, 8))
    periods, amp, avg, ampunc, total_std = result
    msk = periods >= 4
    scat = axes[2].scatter(periods[msk], amp[msk], c=periods[msk], alpha=0.5)
    cbar = plt.colorbar(scat)
    cbar.set_label("Periods")
    # axes[1].colorbar()
    axes[2].set_xlabel("Periods (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_title("Seasonal variation parameters")
    axes[1].errorbar(periods[msk], avg[msk], yerr=total_std[msk], fmt=".", capsize=10)
    axes[1].set_ylim((0, 2))
    axes[1].set_xlabel("Period (s)")
    axes[1].set_ylabel("ZH ratio")
    axes[1].set_title("Extracted ZH ratio")
    filename = join(datadir, "Decomposed", "{}.pdf".format(staid))
    fig.savefig(filename, format="PDF")


def decompose_single_sta(staid, datadir, periods, half_month=3):
    """handle single station condition

    Parameters
    ==========
    staid: str
        station ID
    datadir: str or path-like obj.
        directory of the constrained ZH ratios
    periods: numpy.array
        periods of ZH ratios
    half_month: int
        half length in time when constraining
    """
    # import time serry
    logger.info("Decomposing ZH ratio time series of {}".format(staid))
    starttime, endtime = find_time_duration(staid=staid, datadir=datadir, pat="zh.csv")
    if starttime is np.nan:
        logger.error("No data error for {}".format(staid))
        return

    # Check existence of result file
    filename = join(datadir, "Decomposed", "{}.csv".format(staid))
    if exists(filename):
        logger.info("{} has been handled [ignore !]".format(staid))
        return

    # Get time serry
    months = [dt for dt in rrule(MONTHLY, dtstart=starttime, until=endtime)]

    # import data
    NM, NP = len(months), len(periods)
    timelen = NM - half_month * 2
    if timelen <= 0:
        logger.info("{} has no enough data [ignore !]".format(staid))
        return

    zhs, stds = np.zeros((timelen, NP)), np.zeros((timelen, NP))

    # loop over each seasonal ZH ratio time point
    for idx in range(half_month, NM - half_month):
        # import three months moving average
        sub_months = months[idx - half_month : idx + half_month + 1]
        startt = sub_months[0].strftime("%Y%m")
        endt = sub_months[-1].strftime("%Y%m")
        filename = join(
            datadir, "Constrained/{}_{}/{}.zh.csv".format(startt, endt, staid)
        )
        try:
            pers, zh, std = import_zh(filename)
            # assign period scale
            zhs[idx - half_month, :], stds[idx - half_month, :] = zh, std
        except OSError as err:
            logger.error("NoData for {} [{}]".format(staid, err))
    if (zhs.flatten() == 0).all():
        logger.info("NoData for {} [skipping]".format(staid))
        return

    # loop over each month
    msg = ["# Sine decomposition of time-variated ZH ratios"]
    msg[0] += " data = amp * sine(t / 12 + pha) + avg + eps"
    submsg = [
        "# Period (s)",
        "amp",
        "pha",
        "avg",
        "ampunc",
        "phaunc",
        "avgunc",
        "res_std",
        "total_unc",
    ]
    msg += ["".join(["{:>12}".format(x) for x in submsg])]
    perd_zhs, perd_stds = np.zeros(NM - half_month * 2), np.zeros(NM - half_month * 2)
    for idx in range(NP):
        # each month's ZH ratios
        perd_zhs, perd_stds = zhs[:, idx], stds[:, idx]
        try:
            popt, punc, res_std = curve_fitness(perd_zhs, perd_stds, pers[idx])
            # export curve fitted results
            amp, pha, avg = popt
            ampunc, phaunc, avgunc = punc
            total_unc = res_std + avgunc
            submsg = [
                "{:12.5f}".format(x)
                for x in [
                    pers[idx],
                    amp,
                    pha,
                    avg,
                    ampunc,
                    phaunc,
                    avgunc,
                    res_std,
                    total_unc,
                ]
            ]
            submsg = " ".join(submsg)
            msg.append(submsg)
        except Exception as err:
            logger.info("Unhandled Error [{}]".format(err))
            continue

    # Loop to output result
    subdir = join(datadir, "Decomposed")
    os.makedirs(subdir, exist_ok=True)
    with open(join(subdir, "{}.csv".format(staid)), "w") as f:
        f.write("\n".join(msg))

    # Visulize it
    try:
        pers = np.array([4, 5, 6, 7, 8, 9])
        submonths = months[half_month : NM - half_month]
        visulize_decomposed_result(
            staid=staid,
            datadir=datadir,
            months=submonths,
            rawzhs=zhs,
            pers=pers,
            periods=periods,
        )
        logger.info("Suc. handle ZH ratio of {}".format(staid))
    except Exception as err:
        logger.info("Unhandled Error [{}]".format(err))


if __name__ == "__main__":
    periods = np.concatenate((np.arange(3, 6.0, 0.2), np.arange(6.0, 10.0, 0.5)))
    with open(sys.argv[1]) as f:
        stas = [x.strip().split()[0] for x in f.readlines()]
    half_month = int(sys.argv[2])
    datadir = "./measured.new3"

    # Parallel for constraining the seasonal ZH ratios
    part_func = partial(
        decompose_single_sta, datadir=datadir, periods=periods, half_month=half_month
    )
    NB_PROCESSES = 45
    pool = mp.Pool(NB_PROCESSES)
    traces = pool.starmap(part_func, zip(stas))
    pool.close()
    pool.join()
