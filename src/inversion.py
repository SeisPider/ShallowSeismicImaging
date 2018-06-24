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
import matplotlib.cm as cm

# third-part modulus
import numpy as np

from . import logger
from .model import Model

EARTH_R = 6371
KM2DEG = 111.19

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
        for item in lines[2:]:
            item = item.strip()
            source, obs_theta, pP, P_w, obs_psi, pS, S_w = item.split()
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
    
    def _mcmc_inversion(self, minalpha=0.1, maxalpha=7, minbeta=0.1, maxbeta=5, 
                        maxnum=5000, per=0.1, logfile="Inverted.csv", norm=2):
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
        logger.info("Perform inversion with {:2d}-norm MCMC".format(norm))
        # -------------------------------------------------------------------------
        # Definde likelihood function
        # -------------------------------------------------------------------------
        def likelihood_func(misfit):
            """compute likelihood function based on computed misfit
            """
            return np.exp(-1.0 * misfit)

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
        vpbound=(minalpha, maxalpha); vsbound=(minbeta, maxbeta)
        while not premodel.constrain(vpbound, vsbound):
                premodel = premodel.variation(per=per)
            
        premisfit = premodel.misfit(pPs, pSs, obs_thetas, 
                                    obs_psis, P_ws, S_ws, norm=norm)
        prelikelihood = likelihood_func(premisfit)
        for idx in tqdm(range(maxnum)):
            # perform model variation and determine accept varied model or not
            curmodel = premodel.variation(per=per)   
            while not curmodel.constrain(vpbound, vsbound):
                curmodel = premodel.variation(per=per)
            curmisfit = curmodel.misfit(pPs, pSs, obs_thetas,
                                        obs_psis, P_ws, S_ws, norm=norm)
            curlikelihood = likelihood_func(curmisfit)
    
            # metropolis slection rule 
            Paccept = curlikelihood / prelikelihood
            if rnd.random() < Paccept:
                moveornot = True
            else:
                moveornot = False
    
            # perform differently for this variation
            if moveornot:
                status = {"misfit":curmisfit,
                          "likelihood":curlikelihood}
                curmodel.update_status(status=status)

                # accept model
                acc_models.append(curmodel) 
                likelihoods.append(curlikelihood)

                # reset current model to be previous model
                premodel = curmodel
                prelikelihood = curlikelihood

        self.models = acc_models
        self._statistical_analysis(logfile, minalpha, maxalpha, minbeta, maxbeta, 
                                   pPs, pSs, obs_thetas, obs_psis, P_ws, S_ws)
        
        return acc_models


    def _grid_inversion(self, minalpha, maxalpha, minbeta, maxbeta, maxnum=20,
                        grid=0.05, sample_per=0.8, logfile="Inverted.csv",
                        processor_num=20, norm=2):
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
        logger.info("Perform inversion with {:2d}-norm Grid search".format(norm))
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
        acc_models = pool.starmap(sub_inversion, tqdm(comblist))
        pool.close()
        pool.join()
        
        self.models = acc_models
        self._statistical_analysis(logfile, minalpha, maxalpha, minbeta, maxbeta, 
                                   pPs, pSs, obs_thetas, obs_psis, P_ws, S_ws)
        return acc_models
    
    def _statistical_analysis(self, logfile, minalpha, maxalpha, minbeta, maxbeta,
                              pPs, pSs, obs_thetas, obs_psis, wp, ws):
        """Analysis statistical characteristical results of inversion

        Parameter
        =========
        logfile : str or path-like obj.
            LOG file directory
        minalpha : float
            minimum vp, km/s
        maxalpha : float
            maximum vp, km/s
        minbeta : float
            minimum vs, km/s
        maxbeta : float
            maximum vs, km/s
        obs_thetas : numpy.array
            observed theta (polarization angle from incident P wave)
        obs_psis : numpy.array
            observed psi (polarization angle from incident S wave)
        wp : numpy.array
            weight for thetas in inversion
        ws : numpy.array
            weight for psis in inversion
        """
        # MCMC = "likelihood" in self.models[0].status.keys()

        # -------------------------------------------------------------------------
        # resolve output models and statistically analyse result
        # -------------------------------------------------------------------------
        N = len(self.models)
        alphas = np.zeros(N) 
        betas = np.zeros(N)
        misfits = np.zeros(N)

        for idx, item in enumerate(self.models):
            alphas[idx] = item.alpha
            betas[idx]  = item.beta
            misfits[idx]  = item.status['misfit']

        mean_alpha, mean_beta = alphas.mean(), betas.mean()
        std_alpha, std_beta = alphas.std(), betas.std()

        # -------------------------------------------------------------------------
        # export results into LOG file
        # -------------------------------------------------------------------------
        outmat = np.matrix([alphas, betas, misfits]).T
        np.savetxt(logfile, outmat, fmt="%.5f")
        
        with open(logfile, "a+") as f:
            msg = "\nMean Alpha: {}\n".format(mean_alpha)
            msg+= "STD  Alpha: {}\n".format(std_alpha)
            msg+= "Mean Beta: {}\n".format(mean_beta)
            msg+= "STD  Beta: {}\n".format(std_beta)
            f.write(msg)
        
        # -------------------------------------------------------------------------
        # export statistical result into figure
        # -------------------------------------------------------------------------
        nrows, ncols = 2, 3
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*6, nrows*6))
        # print(axes)
        axes[0,0].hist(alphas, bins=int((maxalpha-minalpha)/0.1), density=True,
                     histtype="stepfilled")
        axes[0,0].set_xlabel("Vp (km/s)")
        axes[0,0].set_ylabel("Probability")
        axes[0,0].set_title(r"$\mu$={:.2f} , $\sigma$={:.2f} (km/s)".format(mean_alpha, std_alpha))

        axes[0,1].hist(betas, bins=int((maxbeta-minbeta)/0.1), density=True,
                     histtype="stepfilled")
        axes[0,1].set_xlabel("Vs (km/s)")
        axes[0,1].set_ylabel("Probability")
        axes[0,1].set_title(r"$\mu$={:.2f}, $\sigma$={:.2f} (km/s)".format(mean_beta, std_beta))

        axes[0,2].plot(misfits, 'o', color="red")
        axes[0,2].set_xlabel("Iteration number")
        axes[0,2].set_ylabel("Misfit")
        axes[0,2].set_title("Misfit variation")

        # -------------------------------------------------------------------------
        # export data fitness into figure
        # -------------------------------------------------------------------------
        model = Model(alpha=mean_alpha, beta=mean_beta)
        thetas, psis = model.syn_theta_psi(pPs, pSs)

        def init(array, value):
            return np.equal(array, np.ones(len(array))*value)
        
        msk = init(wp, np.nan) + init(wp, 0)
        axes[1,0].scatter(pPs[~msk]*KM2DEG, np.rad2deg(obs_thetas[~msk]), c=wp[~msk],
                       alpha=0.5, cmap=cm.Greys, label="Reliable Measurement")
        axes[1,0].plot(pPs[msk]*KM2DEG, np.rad2deg(obs_thetas[msk]), "+",
                       markersize=5, label="Unreliable Measurement")
        axes[1,0].plot(pPs*KM2DEG, np.rad2deg(thetas), "o", color="blue",
                       markersize=5, label="Inverted")
        axes[1,0].set_xlabel(r"Ray Parameter (s/deg)")
        axes[1,0].set_ylabel(r"Measured $\bar\theta$ (deg)")
        axes[1,0].set_title("Measurements Fitness")
        axes[1,0].legend()

        msk = init(ws, np.nan) + init(ws, 0)
        axes[1,1].scatter(pSs[~msk]*KM2DEG, np.rad2deg(obs_psis[~msk]), c=ws[~msk],
                       alpha=0.5, cmap=cm.Greys, label="Reliable Measurement")
        axes[1,1].plot(pSs[msk]*KM2DEG, np.rad2deg(obs_psis[msk]), "+",
                       markersize=5, label="Unreliable Measurement")
        axes[1,1].plot(pSs*KM2DEG, np.rad2deg(psis), "o", color="blue",
                       markersize=5, label="Inverted")
        axes[1,1].set_xlabel(r"Ray Parameter (s/deg)")
        axes[1,1].set_ylabel(r"Measured $\bar\psi$ (deg)")
        axes[1,1].set_title("Measurements Fitness")
        axes[1,1].legend()

        filename = logfile.replace(".LOG", ".INV.png")
        fig.savefig(filename, format="PNG")
        
def sub_inversion(iterid, comb, pPs, pSs, obs_thetas,obs_psis,
                  P_ws, S_ws, sample_per, norm=2):
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
                                          sub_obs_psis, sub_P_ws, sub_S_ws,
                                          norm=norm)
        # print(beta, alpha, misfits[grdidx])
    # search for the minimum misfit
    minvalue = np.nanmin(misfits)
    minidx = np.where(misfits == minvalue)[0][0]
    inv_alpha, inv_beta = comb[minidx]
    
    # output inversion result with dict.
    
    status = { 
                "id"    : iterid,
                "misfit": minvalue
               }
    acc_model = Model(alpha=inv_alpha, beta=inv_beta, status=status)
    return acc_model