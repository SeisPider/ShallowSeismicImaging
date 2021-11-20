# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose: Modulus
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 10:51h, 13/05/2018
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""modulus to handle measurements of polarization direction
"""
# built-in modulus
from os.path import split, join
from copy import deepcopy
import re

# third-part modulus
import numpy as np
from obspy import read
from numpy.linalg import eig, norm

#  self-developed modulus
from . import logger, EARTH_R
from .util import seperate_channels, theoretical_arrival

class Gauger(object):
    """Gauger for measuring polarization direcction
    """
    def __init__(self, event_dir, station_id, model="PREM"):
        """initialization of gauger

        Parameters
        ==========
        event_dir: str or path-like obj.
            event directory contains three components 
            waveforms of this station
        station_id: str
            net.sta
        model: str
            model used to compute ray parameter. It can be
            prem, ak135, iasp91 et al.
        """
        # initialize event info.
        self.eventdir = event_dir
        self.eventid = split(event_dir)[-1]

        self.staid = station_id
        self.model = model

    def __repr__(self):
        return "Gauger for {}@{}".format(self.staid, self.eventdir)

    def Measure_Polar(self, snr_treshd=3, phase="SKS", **kwargs):
        """Measure theta and psi of one event-station pair
    
        Parameter
        =========
        snr_treshd : float
            minimum acceptable SNR
        phase: str
            phase to measure polarizatrion angle
        
        References
        ==========
        Park, Sunyoung, and Miaki Ishii. "Near-surface compressional 
            and shear wave speeds constrained by body-wave polarization 
            analysis." Geophysical Journal International 213.3 (2018): 
            1559-1571.
        """
        logger.info("Preprocessing {}@{}".format(self.staid, self.eventid))
        
        # ----------------------------------------------------------------------
        # Preprocessing waveforms
        # ----------------------------------------------------------------------
        Cutst, p, snr = self.preprocessing(phase=phase, **kwargs)
    
        # ----------------------------------------------------------------------
        # perform PCA to determine polarization angle
        # ----------------------------------------------------------------------
        if Cutst:
            if re.search("P", phase):
                obs_pol, weight = self._polarization_analysis(Cutst)
            elif re.search("S", phase):
                obs_pol, weight = self._polarization_analysis(Cutst, wavetype="S")
        else:
            obs_pol, weight = np.nan, 0

        # ---------------------------------------------------------------------
        # Set the station info.
        # ---------------------------------------------------------------------
        self.sac = Cutst[0].stats.sac

        # ----------------------------------------------------------------------
        # SNR criterion
        # ----------------------------------------------------------------------
        if snr < snr_treshd:
            weight = 0.0
        
        logger.info("Suc. Measuring {}@{}".format(self.staid, self.eventid))
        return obs_pol, p, weight  
      
    def preprocessing(self, win=(-1, 2), noise_win=(5,10), freq_band=None, 
                      desample=None, velo2disp=True, phase="SKS", marker="t1"):
        """Preprocess the three component waveforms
    
        Parameters
        ==========
        win: tuple
            time window refers to the time recorded by marker for measuring
            polarization angle
        noise_win: tuple
            time window refers to the time recorded by marker for measuring
            noise level
        freq_band: tuple
            gives the minimum frequency and maximum one for measurments.
            Default is None, which means don't perform filter to the waveforms
        desample: None or int
            determine to desample the waveform or not. if desample, gives the
            desample factor
        velo2disp: Bool
            determine to whether thansfer the velocity records to displacement 
            ones
        phase: str
            phase to measure polarizatrion angle
        marker: str
            maker to store the phase arrival time, it can be t1-9
        """
        # ----------------------------------------------------------------------
        # impport waveform
        # ----------------------------------------------------------------------
        pattern = join(self.eventdir,".".join([self.staid, "*"]))
        try:
            st = read(pattern)
        except Exception as err:
            logger.error("Unhandled Error [{}]".format(err))
        
        # ------------------------------------------------------------------------
        # If the waveform is velocity, transfer it to displacement
        # ------------------------------------------------------------------------
        if velo2disp:
            st.integrate()

        # -------------------------------------------------------------------------
        # Import model related paramet.
        # -------------------------------------------------------------------------
        model_name = self.model
        
        # -------------------------------------------------------------------------
        # fundamental waveform preprocessing  
        # -------------------------------------------------------------------------
        if desample:
            st.decimate(factor=desample, strict_length=False)
        
        # waveform backup to avoid in-place revision
        Localst = deepcopy(st)
        
        # ----------------------------------------------------------------------
        # filter waveform
        # ----------------------------------------------------------------------
        if freq_band:
            Localst.filter(type="bandpass", freqmin=freq_band[0], 
                           freqmax=freq_band[1])
        
        # ----------------------------------------------------------------------
        # get P and S phase arrivals
        # ----------------------------------------------------------------------
        r, t, z = seperate_channels(Localst, comps=["R", "T", "Z"])
        
        # ----------------------------------------------------------------------
        # get reference ray parameter from 1D model 
        # ----------------------------------------------------------------------
        origin, arrivals = theoretical_arrival(z, model_name, ('ttall', ))
        try:
            if re.search("P", phase):
                reltime = z.stats.sac[marker]
            elif re.search("S", phase):
                reltime = t.stats.sac[marker]
            arr = origin + reltime
        except Exception as err:
            logger.error("No Reliable {} arrival for {}@{}".format(phase, self.staid,
                                                                   self.eventid))
            arr = None
        
        # ----------------------------------------------------------------------
        # Find the phase and shrink time window
        # ----------------------------------------------------------------------
        for idx, item in enumerate(arrivals):
            if item.name == phase:
                arrival = item
                try:
                    right =  arrivals[idx+1].time - item.time
                except: 
                    right = None
                try:
                    left =  arrivals[idx-1].time - item.time
                except: 
                    left = None
                break
    
        
        # replace the time window
        if left:
            trimwin = (np.max([left, win[0]]), win[1])
        
        if right:
            trimwin = (trimwin[0], np.min([right, win[1]]))
        
        p = arrival.ray_param / EARTH_R

        # ----------------------------------------------------------------------
        # trim waveform to obatin isolated phase
        # ----------------------------------------------------------------------
        def trimmer(arr, lwin, subst):
            if arr:
                subst.trim(arr+lwin[0], arr+lwin[1])
            else:
                subst = None
            return subst

        if trimwin[0] > win[1]:
            trimwin = (win[0], trimwin[1])
        
        
        Localst = trimmer(arr, trimwin, Localst)

        if not Localst:
            snr = 0
            return Localst, snr, st
        
        # ----------------------------------------------------------------------
        # construct the Signal-noise-ratio criterion
        # ----------------------------------------------------------------------
        # get noise time window     
        ts = np.array([z.stats.starttime + idx*z.stats.sac.delta
                       for idx in range(len(z.data))])
        
        if arr:
            tn_min, tn_max = arr - noise_win[1], arr - noise_win[0]
            pnidx = np.where(np.logical_and(ts>=tn_min, ts<=tn_max))
            # ------------------------------------------------------------------
            #   compute SNR  
            # ------------------------------------------------------------------
            if re.search("P", phase):
                data = z.data
            elif re.search("S", phase):
                data = r.data

            noise =  data[pnidx]
            snr = np.abs(data).max() / noise.std()
        else:
            snr = 0
        
        return Localst, p, snr


    def _polarization_analysis(self, st, wavetype="P"):
        """Perform polarization analysis
        
        Parameter
        =========
        st : ObsPy.Stream
            The waveforms of an event at particular station. 
        wavetype : str
            specify incident wave type. It can be capital "P" or "S", indicating P
            and S wave, respectively.
        """
        # -------------------------------------------------------------------------
        # construct covariance matrix S = X*X.T/N 
        # X = [q, r], denoting the vertical and radial components, sepearately
        # -------------------------------------------------------------------------
        r, z = seperate_channels(st, comps=["R", "Z"])
         
        XT = np.matrix([z.data - z.data.mean(), r.data - r.data.mean()])
        S = XT * XT.T / XT.shape[1]
    
        # -------------------------------------------------------------------------
        # PCA decomposition and measurement quality assessment
        # -------------------------------------------------------------------------
        w, v = eig(S)                      # w is eigenvalues and v is eigenvectors
        maxidx = w.argmax()
    
        # normalize the first eigenvector
        col1 = v[:,maxidx]
        u1 = col1 / norm(col1)
    
        # for SV system
        if wavetype == "S":
            # As in SAC definition, the Z axis points upward, thus, use -1.0 to
            # change it downward
            App_Pol_Ang =  np.arccos(-1.0 * u1[1])
        else:
            App_Pol_Ang = np.arccos(u1[0])
    
        widx = np.abs(w[maxidx]) / np.abs(w).sum()
        return App_Pol_Ang, widx 
