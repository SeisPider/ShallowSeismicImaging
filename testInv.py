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
if __name__ == '__main__':
    from src.inversion import PolarInv
    Invertor = PolarInv("./YN.PAS.Reg.1S.POL")
    Invertor.inversion(method="mcmc", minalpha=0.1, maxalpha=7, minbeta=0.1, maxbeta=5,
                       logfile="./YN.PAS.Reg.1S.LOG",
                       maxnum=50000, per=0.05, norm=2)
    # print(Invertor.inversion(minalpha=0.1, maxalpha=7, minbeta=0.1, maxbeta=5,
    #                          logfile="./YN.PAS.Reg.1S.LOG",
    #                          maxnum=500, sample_per=0.8, norm=1))