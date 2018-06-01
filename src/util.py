import numpy as np
from obspy.taup import TauPyModel
from obspy.io.sac import SACTrace
from .respider import logger

def rms(x, axis=None):
    """ Function to calculate the root mean square value of an array.
    """
    return np.sqrt(np.mean(x**2, axis=axis))   
    
def rolling_window(a, window):
    """ Efficient rolling statistics with NumPy: This is applied to Picker._statistics() to calculate statistics
        and Summary.threshold() to calcuate threshold to trigger event
        Reference from:
        http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides) 

def theoretical_arrival(tr, modelname="prem", phase_list=["P"]):
    """Get predicted phase arrival based the SAC trace 
    
    Parameter
    =========
    tr : obspy.trace
        obspy trace read from SAC file
    modelname : str
        model name
    phase_list : list
        phase list to get arrivals
    """
    # -------------------------------------------------------------------------
    # construct the origin time
    # -------------------------------------------------------------------------
    sactr = SACTrace.from_obspy_trace(tr)
    evdp, gcarc = sactr.evdp, sactr.gcarc

    # -------------------------------------------------------------------------
    # get waveforms of P and S wave based on given 1D model and time shift
    # -------------------------------------------------------------------------
    model = TauPyModel(model=modelname)
    arr = model.get_travel_times(source_depth_in_km = evdp, 
                                 distance_in_degree = gcarc, 
                                 phase_list=phase_list)
    return  sactr.reftime, arr

def import_stations(stadir):
    """Import all useable stations

    Parameter
    =========
    stadir : str
        directory the station info. file
    """
    logger.info("Loading stations") 
    with open(stadir) as f:
        lines = f.readlines()
        stations = []
        for line in lines:
            stations.append(line.strip().split()[0])
    return stations

def seperate_channels(st, comps=["R", "T", "Z"]):
    """Seperate channels from obspy Stream obj.

    Parameters
    ==========
    st : obspy.Stream
        stream storing all three channels
    comps : list
        channels to be seperated, [RTZ] or [ENZ]
    """
    tr0 = st.select(component=comps[0])[0]
    tr1 = st.select(component=comps[1])[0]
    tr2 = st.select(component=comps[2])[0]
    return tr0, tr1, tr2
