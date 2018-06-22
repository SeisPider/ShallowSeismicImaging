from src.Polarization import Gauger
from src import logger

import numpy as np
import warnings
from glob import glob
from os.path import join
# from tqdm import tqdm

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def single_sta(staid="YN.TOH", datadir="../Data/rotated/", exlog="./YN.TOH.POL",
               phases=["P", "S"], markers=["t1", "t2"], **kwargs):
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
    ps, pols, ws = initarr((len(phases), N)), initarr((len(phases), N)), initarr((len(phases), N))
    
    # Loop over events
    iters = [pols, ps, ws]
    for eidx, event in enumerate(events):
        try:
            for pidx, phase in enumerate(phases):
                Gaug = Gauger(event_dir=event, station_id=staid, model="prem")
                res = Gaug.Measure_Polar(phase=phase, marker=markers[pidx], **kwargs)
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
    single_sta(staid="YN.ZOD", datadir="/home/seispider/Desktop/NearSufaceImaging/Data/Real", 
               exlog="./YN.ZOD.POL", win=(-2, 3), noise_win=(5,10), freq_band=(0.05, 5), desample=None,
               velo2disp=True, phases=("P", "S"), markers=("t1", "t2"))
