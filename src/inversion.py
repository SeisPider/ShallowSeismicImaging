# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 16:43h, 18/04/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""
"""
# built-in modulus
# from copy import deepcopy
from itertools import product
import multiprocessing as mp
from itertools import repeat as rt
from tqdm import tqdm
import random as rnd
import matplotlib.pyplot as plt

# third-part modulus
import numpy as np

from . import logger
from .model import Model

EARTH_R = 6371

class PolarInv(object):
    """Invert surface velocity based on measurements of polarization angles
    """
    def __init__(self, data_dir):
        """initialization of invertion 
        """
        
        self.datadir = data_dir
        self.data = self._import_measurements(data_dir)

    def _import_measurements(self, filedir):
        """Import measurement result
        
        Parameter
        =========
        filedir : str or path-like obj.
            file dirname of the measurements, it's organized by
            file_dirname pP pS obs_thetas obs_psi P_ws S_ws
        """
        logger.info("Importing data from {}".format(filedir))
        with open(filedir, 'r') as f:
            lines = f.readlines()  
        # -------------------------------------------------------------------------
        # resolve each measurement
        # -------------------------------------------------------------------------
        data =  [] 
        for idx, item in enumerate(lines):
            item = item.strip()
            source , pP, pS, obs_theta, obs_psi, P_w, S_w = item.split()
            subdict = {
                        'source': source,
                        'pP':float(pP),
                        'pS':float(pS),
                        'obs_theta':float(obs_theta),
                        'obs_psi':float(obs_psi),
                        'P_w':float(P_w),
                        'S_w':float(S_w)
                      }
            data.append(subdict)     
        return data
    
    def _separate_measurements(self):
        """
        obs_theta : numpy.array
            obaserved P wave apparent polarization angle measured from each event
        obs_psi : numpy.array
            obaserved S wave apparent polarization angle measured from each event
        pPs : numpy.array
            Computed P wave ray parameters each event
        pSs : numpy.array
            Computed S wave ray parameters each event
        P_ws : numpy.array
            P phase weight of each event in inversion
        S_ws : numpy.array
            S phase weight of each event in inversion
        """
        # ----------------------------------------------------------------------
        # Extract measurements from data 
        # ----------------------------------------------------------------------
        N = len(self.data)
        pPs, pSs, obs_thetas = np.zeros(N), np.zeros(N), np.zeros(N)     
        obs_psis, P_ws, S_ws = np.zeros(N), np.zeros(N), np.zeros(N)
        for idx, item in enumerate(self.data):
            pPs[idx] = item['pP']; pSs[idx] = item['pS'];
            obs_thetas[idx] = item['obs_theta']; 
            obs_psis[idx] = item['obs_psi'];
            P_ws[idx] = item['P_w']; S_ws[idx] = item['S_w'];
        return pPs, pSs, obs_thetas, obs_psis, P_ws, S_ws
    
    def inversion(self, method="grid", **kwargs):
        """Choose method to perform inversion 
        """
        if method == "grid":
            return self._grid_inversion(**kwargs)
        if method == "mcmc":
            return self._mcmc_inversion(**kwargs)
    
    def _mcmc_inversion(self, minalpha, maxalpha, minbeta, maxbeta, maxnum=5000,
                        per=0.1, logfile="Inverted.csv"):
        """Perform inversion by marcov chain monte calor method

        Parameter
        =========
        minalpha : float
            minimum vp, km/s
        maxalpha : float
            maximum vp, km/s
        minbeta : float
            minimum vs, km/s
        maxbeta : float
            maximum vs, km/s
        maxnum : int
            maximum number of randomly resampling
        """
        # -------------------------------------------------------------------------
        # Definde likelihood function
        # -------------------------------------------------------------------------
        def likelihood_func(misfit):
            """compute likelihood function based on computed misfit
            """
            return np.exp(-1.0 * misfit / 2)

        # -------------------------------------------------------------------------
        # Obtain observation
        # -------------------------------------------------------------------------
        pPs, pSs, obs_thetas, obs_psis, P_ws, S_ws = self._separate_measurements()

        # -------------------------------------------------------------------------
        # Perform MCMC
        # -------------------------------------------------------------------------
        acc_models, likelihoods = [], []
        # 1. randomly generate initial model
        premodel = Model(alpha=rnd.uniform(minalpha, maxalpha), 
                         beta=rnd.uniform(minbeta, maxbeta))
        for idx in tqdm(range(maxnum)):
            prelikelihood = likelihood_func(premodel.misfit(pPs, pSs, obs_thetas,
                                            obs_psis, P_ws, S_ws))
    
            # perform model variation and determine accept varied model or not
            curmodel = premodel.variation(per=per)
            vpbound=(minalpha, maxalpha); vsbound=(minbeta, maxbeta)
            while not curmodel.constrain(vpbound, vsbound):
                curmodel = premodel.variation(per=per)

            curlikelihood = likelihood_func(curmodel.misfit(pPs, pSs, obs_thetas,
                                            obs_psis, P_ws, S_ws))
    
            # metropolis slection rule 
            Paccept = curlikelihood / prelikelihood
            # if np.isnan(Paccept):
            #     continue
            print(Paccept, curlikelihood)
            if Paccept >= 1:
                moveornot = True
            elif rnd.random() < Paccept:
                moveornot = True
            else:
                moveornot = False
    
            # perform differently for this variation
            if moveornot:
                # accept model
                acc_models.append(curmodel) 
                likelihoods.append(curlikelihood)

                # reset current model to be previous model
                premodel = curmodel
                prelikelihood = curlikelihood
                print("Move")
                # curmodel = premodel.variation(per)
            else:
                print("Stay")
                # curmodel = premodel.variation(per)
        
        # -------------------------------------------------------------------------
        # Statistical analysis
        # -------------------------------------------------------------------------
        alphas, betas = np.zeros(len(acc_models)), np.zeros(len(acc_models))
        for idx, item in enumerate(acc_models):
            alphas[idx] = item.alpha
            betas[idx] = item.beta
        
        mean_alpha, mean_beta = alphas.mean(), betas.mean()
        std_alpha, std_beta = alphas.std(), betas.std()
        
        # -------------------------------------------------------------------------
        # export results into LOG file
        # -------------------------------------------------------------------------
        outmat = np.matrix([alphas, betas, np.array(likelihoods)]).T
        np.savetxt(logfile, outmat, fmt="%.5f")
        
        with open(logfile, "a+") as f:
            msg = "\nMean Alpha: {}\n".format(mean_alpha)
            msg+= "STD  Alpha: {}\n".format(std_alpha)
            msg+= "Mean Beta: {}\n".format(mean_beta)
            msg+= "STD  Beta: {}\n".format(std_beta)
            f.write(msg)
            
        plt.hist(alphas, bins=40)
        plt.show()
        plt.hist(betas, bins=40)
        plt.show()
        return msg


    def _grid_inversion(self, minalpha, maxalpha, minbeta, maxbeta, maxnum=20,
                        grid=0.05, sample_per=0.8, logfile="Inverted.csv",
                        processor_num=20):
        """Perform inversion by grid search and bootstrap
    
        Parameter
        =========
        minalpha : float
            minimum vp, km/s
        maxalpha : float
            maximum vp, km/s
        minbeta : float
            minimum vs, km/s
        maxbeta : float
            maximum vs, km/s
    
        maxnum : int
            maximum number of bootstrap
        grid : float
            velocity step in grid searching
        sample_per : float
            resample of measurements in each iterations
        logfile : str or path-like obj.
            LOG file directory
        processor_num : int
            processor number used in  multiprocessing
        """
        pPs, pSs, obs_thetas, obs_psis, P_ws, S_ws = self._separate_measurements()
    
        # -------------------------------------------------------------------------
        # construct search grid
        # -------------------------------------------------------------------------
        alphas = np.arange(minalpha, maxalpha, grid)
        betas  = np.arange(minbeta, maxbeta, grid)
        comb = list(product(alphas, betas))
         
        # -------------------------------------------------------------------------
        # bootstrap analysis using multiprocess 
        # -------------------------------------------------------------------------
        inv_alphas, inv_betas = np.zeros(maxnum), np.zeros(maxnum)
        minmisfits, iteridx = np.zeros(maxnum), np.arange(maxnum)
        # Perform multiprocessing
        comblist = list(zip(iteridx, rt(comb), rt(pPs), rt(pSs), rt(obs_thetas),
                            rt(obs_psis), rt(P_ws), rt(S_ws), rt(sample_per))) 
        pool = mp.Pool(processor_num)
        outdicts = pool.starmap(sub_inversion, tqdm(comblist))
        pool.close()
        pool.join()
        
        # -------------------------------------------------------------------------
        # resolve output dictionary and statistically analyse result
        # -------------------------------------------------------------------------
        for idx, item in enumerate(outdicts):
            inv_alphas[idx] = item['alpha']
            inv_betas[idx]  = item['beta']
            minmisfits[idx]  = item['misfit']
        mean_alpha, mean_beta = inv_alphas.mean(), inv_betas.mean()
        std_alpha, std_beta = inv_alphas.std(), inv_betas.std()
        
        # -------------------------------------------------------------------------
        # export results into LOG file
        # -------------------------------------------------------------------------
        outmat = np.matrix([inv_alphas, inv_betas, minmisfits]).T
        np.savetxt(logfile, outmat, fmt="%.5f")
        
        with open(logfile, "a+") as f:
            msg = "\nMean Alpha: {}\n".format(mean_alpha)
            msg+= "STD  Alpha: {}\n".format(std_alpha)
            msg+= "Mean Beta: {}\n".format(mean_beta)
            msg+= "STD  Beta: {}\n".format(std_beta)
            f.write(msg)
        inv = {
                 'Mean_Alpha': mean_alpha,
                 'Mean_Beta': mean_beta,
                 'Std_Alpha': std_alpha,
                 'Std_Beta':  std_beta,
               }
        return inv

def sub_inversion(iterid, comb, pPs, pSs, obs_thetas,obs_psis,
                  P_ws, S_ws, sample_per):
    """grid search inversion sub-function
    Parameters
    ==========
    iterid: int
         index of iteration
    comb : list
        list of tuples containing all possible (Vp, Vs) combinations
    CAUTION: all other inputed parameters are the same as that in inversion 
    
    Return
    =======
    inv_alpha : float
        inverted best-fit alpha
    inv_beta : float
        inverted best-fit beta
    minvalue : float
        minimum misfit in grid search 
    """
    # Resample the observations
    obs_num = len(obs_thetas)
    samples_idx = np.array([rnd.randint(0, obs_num-1)
                           for x in range(int(obs_num*sample_per))])
    sub_obs_thetas, sub_obs_psis = obs_thetas[samples_idx], obs_psis[samples_idx]
    sub_P_ws, sub_S_ws = P_ws[samples_idx], S_ws[samples_idx]
    sub_pPs, sub_pSs = pPs[samples_idx], pSs[samples_idx]

    # generate synthetic theta and psi
    misfits = np.zeros(len(comb))
    for grdidx, grdparam in enumerate(comb):
        alpha, beta = grdparam
        
        # energe condiction
        if beta >= np.sqrt(2) * alpha / 2:
            misfits[grdidx] = np.nan
            continue
        # compute ray parameter based on given incident angle
        invmodel = Model(alpha=alpha, beta=beta) 
        misfits[grdidx] = invmodel.misfit(sub_pPs, sub_pSs, sub_obs_thetas, 
                                          sub_obs_psis, sub_P_ws, sub_S_ws)
        # print(beta, alpha, misfits[grdidx])
    # search for the minimum misfit
    minvalue = np.nanmin(misfits)
    minidx = np.where(misfits == minvalue)[0][0]
    inv_alpha, inv_beta = comb[minidx]
    
    # output inversion result with dict.
    outdict = { 
                "id"    : iterid,
                "alpha" : inv_alpha,
                "beta"  : inv_beta,
                "misfit": minvalue
               }
    return outdict