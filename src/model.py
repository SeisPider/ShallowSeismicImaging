# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 20:32h, 01/06/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""Surface velocity model related class
"""
import numpy as np
from copy import deepcopy
import random as rnd

class Model(object):
    def __init__(self, alpha, beta, status=None):
        """initialize the velocity model of

        Parameter
        =========
        alpha : float
            P wave velocity for synthetic generator, km/s
        beta : float
            S wave velocity for synthetic generator, km/s
        """
        self.alpha = alpha
        self.beta = beta
        if status:
            self.status = status
    
    def update_status(self, status):
        """Update status after inversion

        Parameter
        =========
        status: dict.
           update misfit/likelihood status of this particular
           model 
        """
        self.status = status
        
    def syn_theta_psi(self, pPs, pSs):
        """Generate synthetic theta and psi based on given ray parameters
        and vp, vs
    
        Parameter
        =========
        pPs : numpy.array
            P wave ray parameters computed based on 1D model
        pSs  : float
            S wave ray parameters computed based on 1D model

        """
        thetas = 2 * np.arcsin(self.beta * pPs)
        upper = 2 * self.beta**2*pSs * np.sqrt(1 - self.alpha**2*pSs**2)
        lower = self.alpha * (1 - 2*self.beta**2*pSs**2)
            
        psis=  np.arctan2(upper, lower)
        # thetas, psis = np.zeros(len(pPs)), np.zeros(len(pPs))
        # for idx in range(len(pPs)):
        #     pP, pS = pPs[idx], pSs[idx]
        #     thetas[idx] = 2 * np.arcsin(self.beta * pP)
            
        #     upper = 2 * self.beta**2*pS * np.sqrt(1 - self.alpha**2*pS**2)
        #     lower = self.alpha * (1 - 2*self.beta**2*pS**2)
            
        #     psis[idx] =  np.arctan2(upper, lower)
        return thetas, psis

    def misfit(self, pPs, pSs, obs_theta, obs_psi, P_ws, S_ws, norm=2):
        """compute misfit between synthetic polarization direction and observed ones
    
        Parameter
        =========
        pPs : numpy.array
            P wave ray parameters computed based on 1D model
        pSs  : float
            S wave ray parameters computed based on 1D model
        obs_theta : numpy.array
            the obaserved theta measured from each event
        obs_psi : numpy.array
            the obaserved psi measured from each event
        P_ws : numpy.array
            P phase weight of each event in inversion
        S_ws : numpy.array
            S phase weight of each event in inversion
        """
        # -------------------------------------------------------------------------
        # Compute synthetic apparent polarization angle
        # -------------------------------------------------------------------------
        syn_theta, syn_psi = self.syn_theta_psi(pPs, pSs)
        
        # -------------------------------------------------------------------------
        # transfer the rad to deg
        # -------------------------------------------------------------------------
        syn_theta, syn_psi = np.rad2deg(syn_theta), np.rad2deg(syn_psi)
        obs_theta, obs_psi = np.rad2deg(obs_theta), np.rad2deg(obs_psi)
        
        thetadiff, psidiff = np.abs(syn_theta - obs_theta), np.abs(syn_psi - obs_psi) 
        upper = P_ws * thetadiff**norm + S_ws * psidiff**norm
        lower = P_ws + S_ws
        if np.isnan(upper).all():
            return np.nan
        else:
            return np.nansum(upper) / np.nansum(lower)
    
    def variation(self, per=0.1):
            """Randomly generate perturbed model with gaussian distribution and 
            per * value gives sigma
    
            Parameter
            =========
            per : float
                percentage of model parameter variation
            """
            model = deepcopy(self)
            paraidx = np.random.choice(2)
            if paraidx == 0:
                randomwalk = rnd.gauss(0, per) + 1
                model.alpha *= randomwalk 
            elif paraidx == 1:
                randomwalk = rnd.gauss(0, per) + 1
                model.beta  *= randomwalk 
            return model

    def constrain(self, vpbound=(0.1, 7), vsbound=(0.1, 5)):
        """contrain the model physically
        
        Parameter
        =========
        vsbound : tuple
            vs boundary, km/s
        vpbound : tuple
            vp boundary, km/s
        """
        accept = True
        if (self.alpha - vpbound[0]) * (self.alpha - vpbound[1]) > 0:
            accept = False
        if (self.beta - vsbound[0]) * (self.beta - vsbound[1]) > 0:
            accept = False
        # energe conservation
        if self.beta >= np.sqrt(2) * self.alpha / 2:
            accept = False
        return accept