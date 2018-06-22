# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 10:51h, 18/06/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmath import sqrt as csqrt
font = {# 'weight' : 'bold',
        'size'   : 22}
KM2DEG = 111.19

matplotlib.rc('font', **font)

def Measurements(filedir="./YN.PAS.POL", SurfBeta=3.16, SurfAlpha=5.72):
    """visulize measurements of the noisy apparent polaroization angle
    """
    pol, p, weight = np.loadtxt(filedir, unpack=True, usecols=(1,2,3))
    def synpsi(ps, SurfAlpha, SurfBeta):
        upper = 2 * SurfBeta**2 * ps * np.sqrt(1 - SurfAlpha**2*ps**2)
        lower = SurfAlpha * (1 - 2 * SurfBeta**2 * ps**2)
        return np.arctan2(upper, lower)
    # genrate apparent theta and psi
    # thetabar = 2 * np.arcsin(SurfBeta * pP)
    # psibar = np.array([synpsibar(SurfAlpha, SurfBeta, pS[idx], dists[idx]) for idx in range(len(pS))])
    # psibar = np.arctan2(upper, lower)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    def init(array, value):
        return np.equal(array, np.ones(len(array))*value)
    
    msk = init(weight, np.nan) + init(weight, 0)
    ax.plot(p[~msk]*KM2DEG, np.rad2deg(pol[~msk]), "+", markersize=7, label="Reliable Measurement")
    # ax.plot(p[msk]*KM2DEG, np.rad2deg(pol[msk]), "+", label="Unreliable Measurement")
    p.sort()
    ax.plot(p*KM2DEG, np.rad2deg(synpsi(p, SurfAlpha, SurfBeta)), label="Vp=5.8, Vs=3.2 (Real)")
    ax.plot(p*KM2DEG, np.rad2deg(synpsi(p, SurfAlpha-2, SurfBeta)), label="Vp=3.8, Vs=3.2")
    ax.plot(p*KM2DEG, np.rad2deg(synpsi(p, SurfAlpha+2, SurfBeta)), label="Vp=7.8, Vs=3.2")
    ax.plot(p*KM2DEG, np.rad2deg(synpsi(p, SurfAlpha, SurfBeta-2)), label="Vp=5.8, Vs=1.2")
    ax.plot(p*KM2DEG, np.rad2deg(synpsi(p, SurfAlpha, SurfBeta+2)), label="Vp=5.8, Vs=5.2")
    ax.set_xlabel(r"Ray Parameter (s/deg)")
    ax.set_ylabel(r"Measured $\bar\theta$ (deg)")
    ax.set_title("Measurements Fitness of S_waves_SmPvmP_et_al")
    ax.legend()

    plt.show()
    # exportfiledir = filedir.replace(".POL", ".png")
    # fig.savefig(exportfiledir, format="PNG")

if __name__ == '__main__':
    SurfBeta, SurfAlpha = 3.2, 5.8
    Measurements(filedir="./test.POL", SurfBeta=SurfBeta, SurfAlpha=SurfAlpha)