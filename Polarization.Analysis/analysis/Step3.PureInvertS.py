# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose:
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 10:21h, 20/12/2018
#        Usage:
#
#
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
# -------------------------------------------------------------------------------
def import_measurement(logfile):
    """the same as func. name
    """
    # import efficient data
    data = np.loadtxt(logfile, skiprows=2, usecols=(1, 2, 3, 4))
    ang, p, wt = data[:, 0], data[:, 1], data[:, 2]
    msk = ~np.isnan(ang)
    eff_ang, eff_p, eff_wt = ang[msk], p[msk], wt[msk]

    # select data with weight higher than 7
    msk = eff_wt >= 0.7
    eff_ang, eff_p, eff_wt = np.rad2deg(eff_ang[msk]), eff_p[msk], eff_wt[msk]

    # select data where angels are smaller than 80 degree
    msk = eff_ang <= 60
    eff_ang, eff_p, eff_wt = eff_ang[msk], eff_p[msk], eff_wt[msk]
    return eff_ang, eff_p, eff_wt


def syn_ang(meas_p, vs):
    """compute synthetic angle based on given vs and ray 
       parameter
    """
    return np.rad2deg(2 * np.arcsin(vs * meas_p))


def misfit(vs, meas_p, meas_ang, meas_wt):
    """self-defined misfit function
    """
    syn_angs = syn_ang(meas_p, vs)
    upper = ((syn_angs - meas_ang) ** 2 * meas_wt).sum()
    lower = meas_wt.sum()
    return upper / lower


def inversion(staid, meas_p, meas_ang, meas_wt, save=True, bootstrap_num=5000):
    """perform inversion and give uncertainty of vs
    """
    # popt, pcov = curve_fit(syn_ang, meas_p, meas_ang, bounds=(0, 4))
    m = minimize(
        misfit, 3, args=(meas_p, meas_ang, meas_wt), method="SLSQP", bounds=((0, 4.5),)
    )
    mean_vs = m.x[0]

    # Filter the outliers
    residuals = syn_ang(meas_p, mean_vs) - meas_ang
    meanresidual, stdresidual = residuals.mean(), residuals.std()
    msk = np.abs(residuals - meanresidual) <= 2 * stdresidual
    meas_p, meas_ang, meas_wt = meas_p[msk], meas_ang[msk], meas_wt[msk]

    m = minimize(
        misfit,
        mean_vs,
        args=(meas_p, meas_ang, meas_wt),
        method="SLSQP",
        bounds=((0, 4.5),),
    )
    mean_vs = m.x[0]

    # use bootstrap to estimate uncertainty or not
    vss = np.zeros(bootstrap_num)
    measurement_num = len(meas_p)
    for idx in range(bootstrap_num):
        sub_meas_p = choice(meas_p, measurement_num)
        sub_meas_ang = choice(meas_ang, measurement_num)
        sub_meas_wt = choice(meas_wt, measurement_num)
        m = minimize(
            misfit,
            mean_vs,
            args=(sub_meas_p, sub_meas_ang, sub_meas_wt),
            method="SLSQP",
            bounds=((0, 4.5),),
        )
        vss[idx] = m.x[0]
    if save:
        # depict data
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))
        # depict uncertainty
        axes[0].hist(vss, bins=40, density=True, facecolor="g", alpha=0.5)
        axes[0].vlines(
            [mean_vs],
            0,
            1,
            transform=axes[0].get_xaxis_transform(),
            colors="r",
            label="Inverted_Mean",
        )
        boot_mean = vss.mean()
        boot_unc = vss.std()
        axes[0].vlines(
            [boot_mean, boot_mean + boot_unc * 3, boot_mean - boot_unc * 3],
            0,
            1,
            transform=axes[0].get_xaxis_transform(),
            colors="b",
            label="Boot_Mean_Unc",
        )
        axes[0].set_xlabel("Vs (km/s)")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        cax = axes[1].scatter(meas_p, meas_ang, c=meas_wt)
        xarray = np.linspace(meas_p.min(), meas_p.max(), 100)
        axes[1].plot(xarray, syn_ang(xarray, m.x), "r-")
        cbar = fig.colorbar(cax)
        axes[1].set_xlabel("Ray parameter")
        axes[1].set_ylabel("Polarization angle (deg.)")
        axes[1].set_title("Polarization Analysis ({:.3f} km/s)".format(boot_mean))
        axes[1].set_ylim(-40, 90)
        fig.savefig("./MOD/{}.png".format(staid))
        plt.close()
    print("Suc. invert {}".format(staid))
    return vss.mean(), vss.std(), measurement_num


def one_station(logfile):
    """perform inversion and uncertainty analysis of specific station

    Parameters
    ==========
    logfile: str
        path of log file
    """
    try:
        staid = ".".join(basename(logfile).split(".")[0:2])
        meas_ang, meas_p, meas_wt = import_measurement(logfile)
        vs, std, num = inversion(staid, meas_p, meas_ang, meas_wt, save=True)
        msg = "{} {:.5f} {:.5f} {}".format(staid, vs, std, num)
        return msg
    except Exception as err:
        print("Unhandled Error [{}] {}".format(err, staid))


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    from glob import glob
    from os.path import basename
    from numpy.random import choice
    import multiprocessing as mp

    plt.switch_backend("agg")
    NB_PROCESSES = 40
    logfiles = glob("./POL/*.POL")
    msg = ["#staid vs(km/s) vs_unc(km/s) measurement_num"]
    pool = mp.Pool(NB_PROCESSES)
    msgs = pool.starmap(one_station, zip(logfiles))
    msgs = msg + msgs
    outmsg = [x for x in msgs if x is not None]
    # export to inverted log file
    with open("./MOD/test.csv", "w") as f:
        f.write("\n".join(outmsg))
