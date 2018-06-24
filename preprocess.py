# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 13:13h, 25/04/2018
#        Usage: 
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""
"""
# bouild-in modules
from glob import glob
from os.path import join, split, exists 
import os
from copy import deepcopy
import warnings
import multiprocessing as mp
from itertools import repeat as rt

# third-part modules
from obspy import read, UTCDateTime
from obspy.io.sac.sacpz import attach_paz
from obspy.signal.invsim import corn_freq_2_paz

# self-developed modules
from src.respider import SourceResponse, logger
from src.util import import_stations, theoretical_arrival

NB_PROCESSES = 20
warnings.simplefilter(action='ignore', category=FutureWarning)

def removeResponse(tr, freqmin, freqmax, debug=False):
    """
    Remove instrumental response of trace.

    Parameter
    =========
    tr : ObsPy.Trace
        seismic trace
    freqmin : float
        minimum frequency
    freqmax : float
        maximum frequency
    """
    trace =  deepcopy(tr)
    if not (trace):
        return

    network = trace.stats.network
    station = trace.stats.station

    # ============================================
    # Removing instrument response, mean and trend
    # ============================================
    # removing response...
    try:
        trace.detrend(type='constant')
        trace.detrend(type='linear')
        trace.simulate(paz_simulate=corn_freq_2_paz(0.005),
                       remove_sensitivity=True,
                       simulate_sensitivity=True,
                       nfft_pow2=True)
    except Exception as err:
    # cannot preprocess if no instrument response was found,
        # unhandled exception!
        trace = None
        msg = 'Unhandled error: {}'.format(err)
        # printing output (error or ok) message
        logger.error('{}.{} [{}] '.format(network, station, msg))

    # although processing is performed in-place, trace is returned
    # in order to get it back after multi-processing
    return trace

def preprocess_per_station(staid, events, responses_pider, freqmin=0.01, 
                           mindp=60, telemindelta=30, telemaxdelta=90, 
                           win=(-50, 100), utelemindelta=90, utelemaxdelta=140,
                           freqmax=20, exportdir="../Data/Comp_Resp_Rem"):
    """Preprocess of waveforms of paricular station in  events, including 
    removing instrumental response, pick arrivals and apply teleseismic 
    criterion.

    Parameter
    =========
    eventdir : str
        directory of the waveforms
    response_spider : SourceResponse
        class to find the response file for each files
    staid : str
        station id combined with net.sta
    freqmin : float
        minimum frequency
    freqmax : float
        maximum frequency
    mindp : float
        minimum depth of deep eq, in km
    telemindelta : float
        minimum epicentral distance of P,S system, in degree
    telemaxdelta : float
        maximum epicentral distance P,S system, in degree
    utelemindelta : float
        minimum epicentral distance of Pdiff,SKS system, in degree
    utelemaxdelta : float
        maximum epicentral distance of Pdiff,SKS system, in degree
    win : tuple
        time window length to isolate particular phase
    """
    # -------------------------------------------------------------------------
    # Resolve stations in each events
    # -------------------------------------------------------------------------
    net, sta = staid.split(".")
    for event in events:
        
        # Reolve event id and origin time
        eventid = split(event)[-1]
        origin = UTCDateTime(eventid)
        logger.info("Preprocessing {}@{}".format(staid, eventid))
        
        # ---------------------------------------------------------------------
        # check existence of SAC traces 
        # ---------------------------------------------------------------------
        for cha in ["BHZ", "BHE", "BHN"]:
            folderdirname = join(exportdir, eventid)
            # export waveforms
            sacfilename = ".".join([staid, "00", cha, "SAC"])
            wholedir = join(folderdirname, sacfilename)
            if exists(wholedir):
                logger.info("{} Exist Skipping".format(sacfilename))
                continue
        
        try:
            # Check existence of station
            staname = "*.{}.{}.*".format(net, sta)
            st = read(join(event, staname))
        except:
            logger.info("No station in this event !")
            continue
        
        
        # Teleseismic and deep-earthquake criterion
        evdp, gcarc  = st[0].stats.sac.evdp, st[0].stats.sac.gcarc
        if evdp < mindp:
            logger.info("Not deep EQ [skipping]!")
            continue
        if gcarc < telemindelta or gcarc > utelemaxdelta:
            logger.info("Not Required-EQ [skipping]!")
            continue

        # Check if three components are complete
        if len(st) != 3:
            logger.info("Lose component [{}]-[{}]".format(staid, eventid))

        # ---------------------------------------------------------------------
        # Remove instrumental response
        # ---------------------------------------------------------------------
        net_resp = responses_pider.response[net] 
        
        # Attach response for traces
        def construct_trid(trace):
            """same as name
            """
            if trace.stats.location != "":
                trid = trace.id
            else:
                trid = ".".join([trace.stats.network, trace.stats.station,
                                 "00", trace.stats.channel])
            return trid
        
        def trim_waveform(stream, phase, window):
                """trim the raw stream and give back the waveforms of
                the particular phase
                """
                copyst = deepcopy(stream)
                origin, arr = theoretical_arrival(copyst[0], "prem", phase)
                starttime, endtime = origin + window[0], origin + window[1]
                copyst.trim(starttime=starttime, endtime=endtime)
                return copyst

        def get_phase_waveform(stream, exportdir, window, eventid, phase):
            """same as the name
            """
            try:
                cutst = trim_waveform(stream, phase, window)
                for tr in cutst:
                    trid = construct_trid(tr)
                     # Make sure the folder of this station exists
                    folderdirname = join(exportdir, ".".format([phase, "waves"]),
                                         eventid)
                    os.makedirs(folderdirname, exist_ok=True)
                    # export waveforms
                    tr.write(join(folderdirname, trid), format="SAC")
            except Exception as err:
                logger.info("Unhandled Error [{}@{}]-[{}]".format(staid,
                                                           eventid, err))

        try:
            # --------------------------------------------------------------
            # remove response and filter waveforms
            # --------------------------------------------------------------
            for tr in st:
                trid = construct_trid(tr)
                tr_resp = net_resp.responses[trid].get_response(origin)
                # !!!!!!!Caution: Here is velocity 
                attach_paz(tr, tr_resp, tovel=True)
                tr = removeResponse(tr, freqmin=freqmin, freqmax=freqmax)
            
            # --------------------------------------------------------------
            # rotated waveforms
            # --------------------------------------------------------------   
            st.rotate("NE->RT", back_azimuth=st[0].stats.sac.baz)
            
            # --------------------------------------------------------------
            # Handle different phases
            # --------------------------------------------------------------
            if gcarc >= telemindelta and gcarc <= telemaxdelta:
                get_phase_waveform(st, exportdir, win, eventid, ["P"])
                get_phase_waveform(st, exportdir, win, eventid, ["S"])
            elif gcarc >= utelemindelta and gcarc <= utelemaxdelta:
                get_phase_waveform(st, exportdir, win, eventid, ["Pdiff"])
                get_phase_waveform(st, exportdir, win, eventid, ["SKS"])    
            except Exception as err:
                logger.info("Unhandled Error [{}@{}]-[{}]".format(staid,
                                                           eventid, err))
def scan_stations(events, subnet=[]):
    """Select station and perform preprocess for it

    Parameter
    =========
    datadir : str
        directory of the data
    minlat : float 
        minimum latitude
    maxlat : float 
        maximum latitude
    minlon : float 
        minimum longitude
    maxlon : float 
        maximum longitude
    """
    stations = []
    for event in events:
        # Reolve event id and origin time
        traces = glob(join(event, "*.SAC"))
        decompose = lambda x: split(x)[-1].split(".")[-6:-4]
        for trace in traces:
            staid = ".".join(decompose(trace))
            if subnet and staid.split(".")[0] not in subnet:
                continue
            # check the existence in stations
            if staid in stations:
                continue
            # check the location
            tr = read(trace, format="SAC", headonly=True)[0]
            stla, stlo = tr.stats.sac.stla, tr.stats.sac.stlo
            stations.append(staid)
    return stations
    

if __name__ == '__main__':
    # import response files
    sourceresponse = SourceResponse(
        subdir="/home/seispider/Tinyprojects/Response/Response")
    
    # get all event
    pat = "/mnt/data12/China_Data_5.5/2012*"
    events = glob(pat)

    stations = import_stations("./stations.info") 
    # ------------------------------------------------------------------------
    # Perform preprocess for each station 
    # ------------------------------------------------------------------------
    comblist = list(zip(stations, rt(events), rt(sourceresponse)))

    pool = mp.Pool(NB_PROCESSES)
    traces = pool.starmap(preprocess_per_station, comblist)
    pool.close()
    pool.join()
