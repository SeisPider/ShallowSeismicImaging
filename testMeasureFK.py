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

def syn_theta_psi(pP, pS, alpha=5.8, beta=3.36):
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
    theta = 2 * np.arcsin(beta * pP)

    upper = 2 * beta**2*pS * np.sqrt(1 - alpha**2*pS**2)
    lower = alpha * (1 - 2*beta**2*pS**2)

    psi =  np.arctan2(upper, lower)
    return theta, psi


if __name__ == '__main__':
    RAD2DEG = 180 / np.pi  
    Gaug = Gauger(event_dir="/home/seispider/Tinyprojects/fk_examples/ex02/crust100/19950523000231", station_id="YN.PAS",
                  model="ak135")
    # res = Gaug.Measure_Polar_obspy(Pwin=(-1, 1), Swin=(-1, 5), noise_win=(5,10), P_freq_band=None,
    #                                slidlen=4, slidfrac=0.5, S_freq_band=None, desample=None, velo2disp=False)
    res = Gaug.Measure_Polar(Pwin=(-2, 2), Swin=(-2, 2), noise_win=(5,10), P_freq_band=None,
                             S_freq_band=None, desample=None, velo2disp=False, 
                             cakemodelname="taupModPrem.nd")
    obs_theta, obs_psi,  pP, pS, unc_theta, unc_psi = res
    syn_theta, syn_psi = syn_theta_psi(pP, pS, alpha=8.09513, beta=4.48100)
    print(obs_theta * RAD2DEG, syn_theta * RAD2DEG, obs_psi * RAD2DEG, syn_psi * RAD2DEG)
