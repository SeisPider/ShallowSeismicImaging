from src.Polarization import Gauger
from src import logger

import numpy as np
import warnings
from glob import glob
from os.path import join

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def single_sta(staid="YN.TOH", datadir="../Data/rotated/", exlog="./YN.TOH.POL", **kwargs):
    """Perform polarization measurements of single station

    Parameter
    =========
    staid : str
        id of station, net.sta
    datadir : str
        directory of phase picked RTZ component waveforms
    exlog : str
        directory and filename of the log file
    """
    # obtain events list
    events = glob(join(datadir, "*"))
    
    # handle each event
    initarr = lambda x: np.zeros(x)
    N = len(events)
    pPs, pSs, obs_thetas = initarr(N), initarr(N), initarr(N) 
    obs_psis, P_ws, S_ws = initarr(N), initarr(N), initarr(N)

    # Loop over events
    iters = [obs_thetas, obs_psis, pPs, pSs, P_ws, S_ws]
    for eidx, event in enumerate(events):
        try:
            Gaug = Gauger(event_dir=event, station_id=staid)
            res = Gaug.Measure_Polar(**kwargs)

            # res = measurement_per_event(stadirname, event, staid, **kwargs)
            for idx, item in enumerate(iters):
                item[eidx] = res[idx]
        except Exception as err:
            for idx, item in enumerate(iters):
                item[eidx] = np.nan
            logger.error("Unhandled Eror [{}]".format(err))
    
    # define trhe float format
    ff = lambda x: "{:.5f}".format(x)
    with open(exlog, 'w') as f:
        for idx, item in enumerate(events):
            line = "  ".join([item, ff(pPs[idx]), ff(pSs[idx]),
                              ff(obs_thetas[idx]), ff(obs_psis[idx]),
                              ff(P_ws[idx]), ff(S_ws[idx]), "\n"])
            f.writelines(line)

if __name__ == '__main__':
    #Gaug = Gauger(event_dir="../Data/rotated/20141206220510", station_id="YN.TNC")
    #Result = Gaug.Measure_Polar(Plen=2, Slen=5, noise_win=(5,10), 
    #                            P_freq_band=(0.1, 0.7), S_freq_band=(0.05, 0.7))
    #print(Result)
    #obs_theta, obs_psi, pP, pS, Pw, Sw = Result
    # print(np.rad2deg(obs_theta), np.rad2deg(obs_psi))
    single_sta(staid="YN.PAS", datadir="./Data/testWaveforms_1s", 
               exlog="./YN.PAS.Reg.POL.1S", Pwin=(0, 3), Swin=(0, 5), noise_win=(1,3), 
               P_freq_band=None, S_freq_band=None, desample=None, velo2disp=False)
