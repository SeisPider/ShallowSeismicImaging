# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 15:47h, 20/06/2018
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
# -------------------------------------------------------------------------------
"""Construct related utilization

References
==========
1. Brocher, Thomas M. "Empirical relations between elastic wavespeeds and density 
   in the Earth's crust." Bulletin of the seismological Society of America 
   95.6 (2005): 2081-2092.
"""
import numpy as np
from numba import jit


def seperate_channels(st, comps=["R", "T", "Z"]):
    """Seperate channels from obspy Stream obj.

    Parameters
    ==========
    st : obspy.Stream
        stream storing all three channels
    comps : list
        channels to be seperated, [RTZ] or [ENZ]
    """
    trs = []
    for comp in comps:
        trs.append(st.select(component=comp)[0])
    return tuple(trs)


@jit
def Quan_I(freq, AmpN, AmpE, minfreq, maxfreq, grid=0.1, min_ang=0, max_ang=180):
    """Find the RTZ coordinates at this particular time slip
    Parameter
    =========
    freq: numpy.array
        freqeuency range of spectrums
    AmpN: numpy.array
        spectrum of Northern trace
    AmpE: numpy.array
        spectrum of Eastern trace
    minfreq: float
        minimum frequency used to estimated I
    maxfreq: float
        maximum frequency used to estimated I
    grid: float
        degree interval in searching the rotation angle, in degree
    min_ang: float
        minimum rotation angle to start searching, in degree
    max_ang: float
        maximum rotation angle to end searching, in degree
    """
    # Get the search angles
    angles = np.deg2rad(np.arange(min_ang, max_ang, grid))

    # Trim spectrum based on given frequency band
    condition = (freq >= minfreq) * (freq <= maxfreq)
    msk = np.where(condition)
    TAmpN, TAmpE = AmpN[msk], AmpE[msk]

    # Grid search each possible rotation angle
    QuanI = np.array(
        [
            (np.abs(TAmpN * np.cos(ang) + TAmpE * np.sin(ang)) ** 2).sum()
            for ang in angles
        ]
    )
    # Seach angle that maximum the QuanI
    ang_max = angles[QuanI.argmax()]

    # give the rotated spectrum
    maxH = TAmpN * np.cos(ang_max) + TAmpE * np.sin(ang_max)
    minH = TAmpN * np.cos(ang_max + np.pi / 2) + TAmpE * np.sin(ang_max + np.pi / 2)
    return ang_max, maxH, minH, msk
