from src.Polarization import Gauger
from src import logger

import numpy as np
import warnings
from glob import glob
from os.path import join
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat as rt
from functools import partial

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def single_sta(staid, exlog, datadir="../Data/rotated/",
               phases=["P", "S"], markers=["t1", "t2"], snr_treshd=8,
               Pfreq_band=(0.04, 2), Sfreq_band=(0.04, 0.5), **kwargs):
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
    initarr = lambda x: np.ones(x) * np.nan
    N = len(events)
    ps, pols, ws = initarr((len(phases), N)), initarr((len(phases), N)), initarr((len(phases), N))
    
    # Loop over events
    iters = [pols, ps, ws]
    for eidx, event in enumerate(events):
        try:
            for pidx, phase in enumerate(phases):
                Gaug = Gauger(event_dir=event, station_id=staid, model="prem")
                if phase == "P":
                    res = Gaug.Measure_Polar(phase=phase, snr_treshd=snr_treshd, 
                                             marker=markers[pidx], 
                                             freq_band=Pfreq_band, **kwargs)
                elif phase == "S":
                    res = Gaug.Measure_Polar(phase=phase, marker=markers[pidx],
                                             snr_treshd=snr_treshd, 
                                             freq_band=Sfreq_band, **kwargs)
                for idx, item in enumerate(iters):
                    item[pidx, eidx] = res[idx]

        except Exception as err:
            for idx, item in enumerate(iters):
                item[pidx, eidx] = np.nan
            logger.error("Unhandled Eror [{}]".format(err))
        
    
    # define trhe float format
    ff = lambda x: "{:.5f}".format(x)
    with open(exlog, 'w') as f:
        msg  = "# Polarization measurements of {} at {} phase\n".format(staid, ".".join(phases))
        msg += "# Directory           polarization_angle(rad)  ray_parameter(s/km)  weight\n"
        f.write(msg)
        for idx, item in enumerate(events):
            parts = []
            parts.append(item)
            for pidx in range(len(phases)):
                parts.append(ff(pols[pidx, idx]))
                parts.append(ff(ps[pidx, idx]))
                parts.append(ff(ws[pidx, idx]))
            parts.append("\n")
            line = "  ".join(parts)
            f.writelines(line)

if __name__ == '__main__':
    NB_PROCESSES = 10
    with open("./Data/GS.net") as f:
        staids = [item.strip() for item in f.readlines()]
        logfiles = [".".join([item, "POL"]) for item in staids]
    
    datadir = "/home/seispider/Desktop/NearSufaceImaging/Data/Real"

    pool = mp.Pool(NB_PROCESSES)
    part_m = partial(single_sta, datadir="/home/seispider/Desktop/NearSufaceImaging/Data/Real", 
                                 win=(-1, 4), noise_win=(5,10), Pfreq_band=(0.04, 2),
                                 Sfreq_band=(0.02, 0.3), snr_treshd=8,
                                 desample=None, velo2disp=True, phases=("P", "S"), 
                                 markers=("t1", "t2"))
    results = pool.starmap(part_m, zip(staids, logfiles))
