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

def syn_theta_psi(pP, pS, alpha=5.8, beta=3.2):
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
    Gaug = Gauger(event_dir="./Data/testWaveforms/19950523165148", station_id="YN.PAS")
    res = Gaug.Measure_Polar(Pwin=(0, 2), Swin=(0, 5), noise_win=(5,10), P_freq_band=None,
                             S_freq_band=None, desample=None, velo2disp=False)
    obs_theta, obs_psi,  pP, pS, Pw, Sw = res
    syn_theta, syn_psi = syn_theta_psi(pP, pS, alpha=5.8, beta=3.2)
    print(obs_theta, syn_theta, obs_psi, syn_psi)
