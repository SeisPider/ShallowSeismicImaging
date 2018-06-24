# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 19:00h, 15/05/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""
"""
import multiprocessing as mp
from src.inversion import PolarInv
from functools import partial

def single_inv(staid, **kwargs):
    """Perform surface velocity inversion for measurement of particular station 
    """
    Invertor = PolarInv("./{}.POL".format(staid))
    Invertor.inversion(logfile="./{}.LOG".format(staid),**kwargs)
    return None

if __name__ == '__main__':
    NB_PROCESSES = 24 
    with open("./inv.log") as f:
        staids = [item.strip() for item in f.readlines()]

    pool = mp.Pool(NB_PROCESSES)
    part_m = partial(single_inv, method="mcmc", minalpha=0.1, maxalpha=7, 
                                 minbeta=0.1, maxbeta=6, maxnum=50000, 
                                 per=0.05, norm=1)
    results = pool.starmap(part_m, zip(staids))
