# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 17:13h, 16/05/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from src.Polarization import Gauger

def syn_theta_psi(p, alpha=5.8, beta=3.36, phase="P"):
    """Generate synthetic theta and psi based on given ray parameters
    and vp, vs

    Parameter
    =========
    pPs : numpy.array
        P wave ray parameters computed based on 1D model
    pSs  : float
        S wave ray parameters computed based on 1D model
    alpha : float
        P wave velocity for synthetic generator, km/s
    beta : float
        S wave velocity for synthetic generator, km/s
    """
    if phase == "P":
        theta = 2 * np.arcsin(beta * p)
        return theta
    else:
        upper = 2 * beta**2 * p * np.sqrt(1 - alpha**2*p**2)
        lower = alpha * (1 - 2 * beta**2 * p**2)
        # upper = 2 * beta**2*p * np.sqrt(1 - alpha**2*p**2)
        # lower = alpha * (1 - 2*beta**2*p**2)
        return np.arctan2(upper, lower)
       

if __name__ == '__main__':
    RAD2DEG = 180 / np.pi  
    Gaug = Gauger(event_dir="/home/seispider/Desktop/NearSufaceImaging/Data/IRIS/test/IU.ANMO.0000", station_id="IU.ANMO",
                  model="ak135")
    # res = Gaug.Measure_Polar_obspy(Pwin=(-1, 1), Swin=(-1, 5), noise_win=(5,10), P_freq_band=None,
    #                                slidlen=4, slidfrac=0.5, S_freq_band=None, desample=None, velo2disp=False)
    res = Gaug.Measure_Polar(win=(0, 6), noise_win=(5,10), freq_band=(0.04, 0.5),
                             desample=None, velo2disp=False, phase="SKS", marker="t3")
    obs_pol, p, weight = res
    syn_psi = syn_theta_psi(p, alpha=5.8, beta=3.36, phase="SKS")
    print(obs_pol * RAD2DEG, syn_psi * RAD2DEG)
