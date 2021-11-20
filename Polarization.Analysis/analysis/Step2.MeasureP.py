from src.Polarization import Gauger
from src import logger

import numpy as np
import warnings
from glob import glob
from os.path import join
import multiprocessing as mp
from functools import partial

# ignore warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def single_sta(
    staid,
    exlog,
    datadir="../Data/rotated/",
    phase="P",
    marker="t1",
    snr_treshd=8,
    freq_band=(0.04, 2),
    **kwargs
):
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
    ps, pols, ws = initarr(N), initarr(N), initarr(N)
    bazs = initarr(N)

    # Loop over events
    iters = [pols, ps, ws]
    for eidx, event in enumerate(events):
        try:
            Gaug = Gauger(event_dir=event, station_id=staid, model="prem")

            res = Gaug.Measure_Polar(
                phase=phase,
                snr_treshd=snr_treshd,
                marker=marker,
                freq_band=freq_band,
                **kwargs
            )
            for idx, item in enumerate(iters):
                item[eidx] = res[idx]
            bazs[eidx] = Gaug.sac.baz

        except UnboundLocalError:
            logger.error("No data for {}@{}".format(staid, event))
        except Exception as err:
            for idx, item in enumerate(iters):
                item[eidx] = np.nan
            bazs[eidx] = np.nan
            logger.error("Unhandled Eror [{}]".format(err))
    # define the print format
    ff = lambda x: "{:.5f}".format(x)
    with open(exlog, "w") as f:
        msg = "# Polarization measurements of {} at {} phase\n".format(staid, phase)
        msg += "# Directory  polarization_angle(rad)  ray_parameter(s/km)  weight  back_azimuth (deg)\n"
        f.write(msg)
        for idx, item in enumerate(events):
            parts = []
            parts.append(item)
            parts.append(ff(pols[idx]))
            parts.append(ff(ps[idx]))
            parts.append(ff(ws[idx]))
            parts.append(ff(bazs[idx]))
            parts.append("\n")
            line = "  ".join(parts)
            f.writelines(line)


if __name__ == "__main__":
    NB_PROCESSES = 40
    with open("./info/test.net") as f:
        staids = [item.strip().split()[0] for item in f.readlines()]
        logfiles = ["./POL/" + ".".join([item, "P", "POL"]) for item in staids]
    pool = mp.Pool(NB_PROCESSES)
    part_m = partial(
        single_sta,
        datadir="../data",
        win=(-1, 4),
        noise_win=(5, 10),
        freq_band=(0.04, 2),
        snr_treshd=3,
        desample=None,
        velo2disp=False,
        phase="P",
        marker="t5",
    )
    results = pool.starmap(part_m, zip(staids, logfiles))
