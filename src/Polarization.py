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

# import matplotlib.pyplot as plt
from obspy.signal.polarization import polarization_analysis

# self-developed part
from . import logger
from .util import seperate_channels, theoretical_arrival, cake_arr

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
        self.model = self.__model_init(model)

    def __repr__(self):
        return "Gauger for {}@{}".format(self.staid, self.eventdir)

    def __model_init(self, model_name):
        """Initialize 1D model used to compute ray parameter
        """
        if model_name.lower() == "prem":
            # ------------------------------------------------------------------
            # Use prem model
            # ------------------------------------------------------------------
            model = {
                      'name'     : "prem",
                      'SurfAlpha': 5.8,
                      'SurfBeta' : 3.2
                    }
        elif model_name.lower() == "ak135":
            # ------------------------------------------------------------------
            # Use ak135 model
            # ------------------------------------------------------------------
            model = {
                      'name'     : "ak135",
                      'SurfAlpha': 5.8,
                      'SurfBeta' : 3.2
                    }
        elif model_name.lower() == "iasp91":
            # ------------------------------------------------------------------
            # Use ak135 model
            # ------------------------------------------------------------------
            model = {
                      'name'     : "iasp91",
                      'SurfAlpha': 5.8,
                      'SurfBeta' : 3.360
                    }
        elif model_name.lower() == "pwdk":
            # ------------------------------------------------------------------
            # Use pwdk model
            # ------------------------------------------------------------------
            model = {
                      'name'     : "pwdk",
                      'SurfAlpha': 5.8,
                      'SurfBeta' : 3.350
                    }
        else:
            raise Exception("Invalid model name:{} [Use prem/ak135 instead]".format(model_name))
        return model

    def Measure_Polar(self, snr_treshd=3, **kwargs):
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
            
        Pst, Sst, pP, pS, Psnr, Ssnr, _, _, _ = self.preprocessing(**kwargs)
    
        # ----------------------------------------------------------------------
        # perform PCA
        # ----------------------------------------------------------------------
        if Pst:
            obs_theta, Pw = self._polarization_analysis(Pst)
        else:
            obs_theta, Pw = np.nan, 0
        
        if Sst:
            obs_psi, Sw = self._polarization_analysis(Sst, wavetype="S")
        else:
            obs_psi, Sw = np.nan, 0

        # ---------------------------------------------------------------------
        # Set the station info.
        # ---------------------------------------------------------------------
        self.sac = Pst[0].stats.sac

        # ----------------------------------------------------------------------
        # SNR criterion
        # ----------------------------------------------------------------------
        if Psnr < snr_treshd:
            Pw = 0.0
        if Ssnr < snr_treshd:
            Sw = 0.0
        
        logger.info("Suc. Measuring {}@{}".format(self.staid, self.eventid))
        return obs_theta, obs_psi, pP, pS, Pw, Sw  
      
    def preprocessing(self, win=(-1, 2), Swin=(-1,4), noise_win=(5,10), freq_band=None,
                      S_freq_band=None, desample=None, velo2disp=True, 
                      phase="SKS"):
        """Preprocess the three component waveforms
    
        Parameter
        =========
        st : ObsPy.Stream
            The waveforms of an event at particular station. 
            The first trace is BHZ, then BHN and last the BHE
        model :dict
            1D model related properties, including
            name : name of reference 1D model used in TauP, e.g. PREM
            SurfAlpha : surface P wave velocity of 1D model, e.g. 5.8 km/s
            SurfBeta  : surface S wave velocity of 1D model, e.g. 3.2 km/s
        Plen : float
            length of time window enclosing the P phase, e.g. Plen=3 indicates 
            the program cut P wave waveform during (Parr-Plen/2, Parr+Plen/2).
        Slen : float
            length of time window enclosing the S phase, e.g. Slen=3 indicates 
            the program cut S wave waveform during (Sarr, Sarr+Slen).
        noise_win : tuple
            noise window ahead from the phase arrival, e.g. (5, 10) means using 
            waveform 5 to 10 second ahead from the phase arrival to define the SNR
        P_freq_band : tuple
            give freqeuency band of P wave with (fp_min, fp_max)
        S_freq_band : tuple
            give freqeuency band of S wave with (fs_min, fs_max)
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
        # Import model related paramet
        # -------------------------------------------------------------------------
        model_name, SurfAlpha = self.model['name'], self.model['SurfAlpha'],
        # SurfBeta = self.model['SurfBeta']
    
        # -------------------------------------------------------------------------
        # construct the origin time
        # -------------------------------------------------------------------------
        sach = (st[0]).stats.sac     # demo sac header 
        nzyear, nzjday, nzhour = sach.nzyear, sach.nzjday, sach.nzhour 
        nzmin,  nzsec,  nzmsec = sach.nzmin,  sach.nzsec,  sach.nzmsec
        dt = sach.delta 
        origin = UTCDateTime(year=nzyear, julday=nzjday, hour=nzhour, minute=nzmin,
                             second=nzsec, microsecond=nzmsec)
        
        # -------------------------------------------------------------------------
        # fundamental waveform preprocessing  
        # -------------------------------------------------------------------------
        if desample:
            st.decimate(factor=desample, strict_length=False)
        
        # waveform backup to avoid in-place revision
        Pst, Sst = deepcopy(st), deepcopy(st)
        
        # waveform filter
        if P_freq_band:
            Pst.filter(type="bandpass", freqmin=P_freq_band[0],
                       freqmax=P_freq_band[1], zerophase=True)
        if S_freq_band:
            Sst.filter(type="bandpass", freqmin=S_freq_band[0], 
                       freqmax=S_freq_band[1], zerophase=True)

        # ----------------------------------------------------------------------
        # get P and S phase arrivals
        # ----------------------------------------------------------------------
        try:
            Pr, _, Pz = seperate_channels(Pst, comps=["R", "T", "Z"])
            Parr = origin + Pz.stats.sac.t1
        except Exception as err:
            logger.error("No Reliable P arrival for {}@{}".format(self.staid,
                                                                  self.eventid))
            Parr = None
        
        try:
            Sr, St, Sz = seperate_channels(Sst, comps=["R", "T", "Z"])
            Sarr = origin + St.stats.sac.t2
        except Exception as err:
            logger.error("No Reliable S arrival for {}@{}".format(self.staid,
                                                                  self.eventid))
            Sarr = None
        # print(Sz.plot())
        
        
        if not cakemodelname:
            # ----------------------------------------------------------------------
            # get reference ray parameter from 1D model 
            # ----------------------------------------------------------------------
            _, Ptrav = theoretical_arrival(Pz, modelname=model_name, phase_list=["P"])
            _, Stravs = theoretical_arrival(Sz, modelname=model_name, phase_list=('ttall', ))
            
            # find the S phase
            # print(Stravs)
            for idx, item in enumerate(Stravs):
                if item.name == "S":
                    Strav = item
                    try:
                        right =  Stravs[idx+1].time - item.time
                        # print(Stravs[idx+1].name)
                        # print(right)
                    except: 
                        right = None
                    
                    try:
                        left =  Stravs[idx-1].time - item.time
                    except: 
                        left = None
                    break
    
            # replace the time window
            if left:
                lftbound = np.max([left, Swin[0]])
                Strimwin = (lftbound, Swin[1])
            
            if right:
                rightbound = np.min([right, Swin[1]])
                Strimwin = (Strimwin[0], rightbound)
            Swin = Strimwin
            
            incP, incS = np.deg2rad(Ptrav[0].incident_angle), np.deg2rad(Strav.incident_angle)
            pP, pS = np.sin(incP) / SurfAlpha, np.sin(incS) / SurfBeta
        else:
            print("Use cake for ray tracing !")
            _, pP = cake_arr(Pz, modelname=cakemodelname, phase="P")
            _, pS = cake_arr(Pz, modelname=cakemodelname, phase="S")

        # pP, pS = Ptrav[0].incident_angle
        
        # # Determine P and S wave
        # Parr, Sarr = origin + Ptrav.time, origin + Strav.time
        # try:
        #     trav_P = rawst[0].stats.sac.t8
        #     if Ptrav != -12345:
        #         Parr = origin + trav_P
        # except:
        #     print("No P wave Time picked, Use PREM predicted [{}]".format(rawst[0].id))
        #     print("                         [{}]".format(origin))
        
        # try:
        #     trav_S = rawst[0].stats.sac.t9
        #     if Strav != -12345:
        #         Sarr = origin + trav_S
        # except:
        #     print("No S wave Time picked, Use PREM predicted [{}]".format(rawst[0].id))
        #     print("                          [{}]".format(origin))

        # ----------------------------------------------------------------------
        # trim P and S phases
        # ----------------------------------------------------------------------
        def trimmer(arr, lwin, subst):
            if arr:
                subst.trim(arr+lwin[0], arr+lwin[1]) 
                # print(arr, lwin, st)
            else:
                subst = None
            return subst 
        # trimlen = np.min([Swin[1], ScS_S])
        # Swin = (Swin[0], trimlen)
        # print(Swin)
        Pst = trimmer(Parr, Pwin, Pst)
        Pwindt =  Parr + Pwin[0], Parr + Pwin[1] 
        Sst = trimmer(Sarr, Swin, Sst)
        Swindt =  Sarr + Swin[0], Sarr + Swin[1] 

        # print("{} for {}".format(trimlen, pS))
        # if Parr:
        #     Pst.trim(starttime=Parr, endtime=Parr+Plen)
        # else:
        #     Pst = None
        
        # if Sarr:
        #     Sst.trim(starttime=Sarr, endtime=Sarr+Slen)
        # else:
        #     Sst = None

        if not Pst and not Sst:
            Psnr = Ssnr = 0
            return Pst, Sst, pP, pS, Psnr, Ssnr, Pwindt, Swindt, st
        
        # if Strimwin[1] < Swin[1]:
        #     Psnr = Ssnr = 0
        #     return Pst, Sst, pP, pS, Psnr, Ssnr, Pwindt, Swindt, st
        
        # ----------------------------------------------------------------------
        # ensure means of waveforms to be zero
        # ----------------------------------------------------------------------
        # for st in [Pst, Sst]:
        #     if not st:
        #         continue
            # st.detrend(type="linear")
        
        # ----------------------------------------------------------------------
        # construct the Signal-noise-ratio criterion
        # ----------------------------------------------------------------------
        # get noise time window
        ts = np.array([Pz.stats.starttime + idx*dt
                       for idx in range(len(Pz.data))])
        if Parr:
            Ptn_min, Ptn_max = Parr - noise_win[1], Parr - noise_win[0]
            pnidx = np.where(np.logical_and(ts>=Ptn_min, ts<=Ptn_max))
            # ------------------------------------------------------------------
            #   compute P wave SNR  
            # ------------------------------------------------------------------
            Pnoise =  Pz.data[pnidx]
            Psnr = np.abs(Pst[0].data).max() / Pnoise.std()
        else:
            Psnr = 0
        
        if Sarr:
            Stn_min, Stn_max = Sarr - noise_win[1], Sarr - noise_win[0]
            snidx = np.where(np.logical_and(ts>=Stn_min, ts<=Stn_max))
            Snoise = Pz.data[snidx]
            # ------------------------------------------------------------------
            #   compute S wave SNR  
            # ------------------------------------------------------------------
            Ssnr = np.abs(Sst[0].data).max() / Snoise.std()
        else:
            Ssnr = 0
        return Pst, Sst, pP, pS, Psnr, Ssnr, Pwindt, Swindt, st


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
        # st.plot()
        zdata = st[2].data
        rdata = st[0].data
        XT = np.matrix([zdata - zdata.mean(), rdata - rdata.mean()])
        S = XT * XT.T / XT.shape[1]
    
        # -------------------------------------------------------------------------
        # PCA decomposition and measurement quality assessment
        # -------------------------------------------------------------------------
        w, v = eig(S)     # w is eigenvalues and v is eigenvectors
        maxidx = w.argmax()
    
        # normalize the first eigenvector
        col1 = v[:,maxidx]
        u1 = col1 / norm(col1)
        # print(u1)
        
        # print(col1, u1)
        # for SV system
        # print(u1)
        if wavetype == "S":
            # As in SAC definition, the Z axis points upward, thus, use -1.0 to
            # change it downward
            print(u1)
            App_Pol_Ang =  np.arccos(np.abs(u1[1]))
        else:
            # App_Pol_Ang = np.arctan2(u1[1], u1[0])
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
            
