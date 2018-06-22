
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 10:51h, 13/05/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""modulus to handle measurements of polarization direction
"""
from os.path import split, join
from copy import deepcopy


# third-part modulus
import numpy as np
from obspy import read, UTCDateTime 
from numpy.linalg import eig, norm
import re

# import matplotlib.pyplot as plt
from obspy.signal.polarization import polarization_analysis

# self-developed part
from . import logger, EARTH_R
from .util import seperate_channels, theoretical_arrival

class Gauger(object):
    """Gauger for measuring polarization direcction
    """
    def __init__(self, event_dir, station_id, model="PREM"):
        """initialize of gauger
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
    
        Parameter
        =========
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
        r, t, z = seperate_channels(Localst, comps=["R","T", "Z"])
        
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
            App_Pol_Ang =  np.arccos(np.abs(u1[1]))
        else:
            App_Pol_Ang = np.arccos(u1[0])
    
        # ----------------------------------------------------------------------
        # estimate interpolation degree index
        # ----------------------------------------------------------------------
        if App_Pol_Ang >= np.pi / 2:
            widx = 0
        else:
            widx = np.real(w[maxidx] / w.sum())

        return App_Pol_Ang, widx
    
    def Measure_Polar_obspy(self, snr_treshd=3, slidlen=2, slidfrac=0.1, minfreq=0.001, maxfreq=None, 
                            **kwargs):
        """Measure theta and psi of one event-station pair
    
        Parameter
        =========
        datadir : str
            directory of data of this staid and eventid
        staid : str
            station ID net.sta
        eventid : str
            event ID
        snr_treshd : float
            minimum acceptable SNR
        """
        logger.info("Preprocessing {}@{}".format(self.staid, self.eventid))
            
        Pst, Sst, pP, pS, Psnr, Ssnr, Pwindt, Swindt, st = self.preprocessing(**kwargs)
    
        # ----------------------------------------------------------------------
        # Measure the polarization angle
        # ----------------------------------------------------------------------
        if not maxfreq:
            # use nyquist frequency
            maxfreq = 1.0 / (2 * st[0].stats.sampling_rate)

        if Pst:
            # this choice of azimuth is arbitary
            st.rotate("RT->NE", back_azimuth=30)
            starttime, endtime = Pwindt
            result = polarization_analysis(st, slidlen, slidfrac, minfreq, maxfreq,
                                           stime=starttime, etime=endtime, verbose=False, 
                                           method='pm', var_noise=0.0)
            incidence, inc_uncertainty = result['incidence'], result['incidence_error']
            print(incidence)
            minidx = np.array(incidence).argmin()
            obs_theta   = np.deg2rad(incidence[minidx])
            delta_theta = np.deg2rad(inc_uncertainty[minidx])
        else:
            obs_theta, delta_theta = np.nan, np.nan
        
        if Sst:
            st.rotate("RT->NE", back_azimuth=30)
            starttime, endtime = Swindt
            result = polarization_analysis(st, slidlen, slidfrac, minfreq, maxfreq,
                                           stime=starttime, etime=endtime, verbose=False, 
                                           method='pm', var_noise=0.0)
            incidence, inc_uncertainty = result['incidence'], result['incidence_error']
            print(90 - incidence)
            maxidx = (90 - np.array(incidence)).argmax()
            obs_psi   = np.deg2rad(90 - incidence[maxidx])
            delta_psi = np.deg2rad(inc_uncertainty[maxidx])
        else:
            obs_psi, delta_psi = np.nan, np.nan

        # ----------------------------------------------------------------------
        # SNR criterion
        # ----------------------------------------------------------------------
        if Psnr < snr_treshd:
            delta_theta = 0.0
        if Ssnr < snr_treshd:
            delta_psi = 0.0

        logger.info("Suc. Measuring {}@{}".format(self.staid, self.eventid))
        return obs_theta, obs_psi, pP, pS, delta_theta, delta_psi  
            
