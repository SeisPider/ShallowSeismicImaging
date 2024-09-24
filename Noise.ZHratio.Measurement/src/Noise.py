# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
#      Purpose: Modulus
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 14:10h, 04/07/2018
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2018 Xiao Xiao
# -------------------------------------------------------------------------------
"""Handle the measurements of ZH ratio from seismic noise
"""
import numpy as np
from .utils import seperate_channels, Quan_I
from scipy.fftpack import fft, fftfreq
from . import logger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from sklearn.linear_model import LinearRegression
import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from os.path import join
import os
import re
import portion as P
from obspy import Stream

# plt.switch_backend("agg")
MINVALUE = 10 ** (-3)


class Noise(object):
    """Class to handle sseismic noise at"""

    def __init__(self, stream=None, starttime=None, endtime=None, staid=None):
        """Class to handle continous waveforms"""
        if stream:
            self.stream = stream
            # Get info. from particular trace
            tr = stream[0]
            self.staid = ".".join(tr.id.split(".")[0:2])

            # Get location info.
            # sachr = tr.stats.sac
            # stlo, stla, stel = sachr["stlo"], sachr["stla"], sachr["stel"]
            # self.coord = (stlo, stla, stel)

            # Get starttime and endtime

            self.starttimes = np.array([x.stats.starttime for x in stream])
            self.starttime = self.starttimes.min()
            self.endtimes = np.array([x.stats.endtime for x in stream])
            self.endtime = self.endtimes.max()

            # Get gaps
            gaps = stream.get_gaps()
            self.gaps = [
                P.closed(x[4], x[5])
                for x in gaps
                if re.search("Z", x[3]) and x[5] - x[4] >= 60
            ]
        else:
            self.staid = staid
            self.starttime, self.endtime = starttime, endtime

    def CutSlices(self, timelen=3600, step_rate=0.5, nprocessor=30):
        """Cut the continous waveform into enumerous subwavforms

        Parameters
        ==========
        timelen: float
            time lenth of each slices, in second
        step_rate: float
            overlap ratio for slicing continous waveform, it's inverse should
            be integer
        """
        logger.info("Cut waveforms into slices with {:.5f} s".format(timelen))

        # Create critical time points for slices cutting
        timepoints = np.arange(
            self.starttime, self.endtime, timelen * step_rate
        ).tolist()

        # Cut waveforms
        self.SliceWaves = []
        step = int(1.0 / step_rate)
        idxs = np.arange(len(timepoints))[:-step]

        if len(self.gaps) != 0:
            part_func = partial(
                sub_slice,
                timepoints=timepoints,
                gaps=self.gaps,
                step=step,
                timelen=timelen,
            )
            pool = mp.Pool(nprocessor)
            judge_results = pool.starmap(part_func, zip(idxs))
            pool.close()
            pool.join()

            self.SliceWaves = [x for x in judge_results if x]
        else:
            self.SliceWaves = []
            for idx in idxs:
                startt, endt = timepoints[idx], timepoints[idx + step]
                self.SliceWaves.append(NoiseSlice(startt, endt))
        # Suc. info.
        logger.info("NB of segments: {}".format(len(self.SliceWaves)))
        logger.info("Suc. slice waveforms")

    def Measure_ratios(
        self, periods, freq_grid=0.01, nprocessor=1, startup="fork", **kwargs
    ):
        """
        Parameters
        ==========
        periods: numpy.array
            period band for ZH reatio computation
        period_grid: float
            period band width
        nprocessor: int
            processor number used for computation
        startup: str.
            method to start up the multiprocessing, fork or spawn
            fork requires larger memory and is more time efficient, while spawn is
            to the opposite

        kwargs: variable args
            arguments for quant. I measurements
        """
        # Discretize the frequency band
        self.periods, self.freq_grid = periods, freq_grid

        # Measure ratios for each slices
        if nprocessor == 1:
            ZHratios, HTratios, DPHIs, AZs = [], [], [], []
            for item in tqdm(self.SliceWaves):
                try:
                    ZH, HT, DPHI, AZ = item.Measure_ZH(
                        rawst=self.stream,
                        periods=periods,
                        freq_grid=freq_grid,
                        **kwargs
                    )
                    ZHratios.append(ZH)
                    HTratios.append(HT)
                    DPHIs.append(DPHI)
                    AZs.append(AZ)
                except ValueError:
                    continue
        elif nprocessor > 1:
            # stream = self.stream
            # starttimes = self.starttimes
            # endtimes = self.endtimes
            partial_func = partial(
                sub_func,
                rawst=self.stream,
                endtimes=self.endtimes,
                starttimes=self.starttimes,
                periods=periods,
                freq_grid=freq_grid,
                **kwargs
            )
            pool = mp.get_context(startup).Pool(nprocessor)
            measurements = pool.starmap(partial_func, zip(self.SliceWaves))
            pool.close()
            pool.join()

            # Seperate results
            ZHratios, HTratios, DPHIs, AZs = [], [], [], []
            for measurement in measurements:
                ZH, HT, DPHI, AZ = measurement
                if ZH is None:
                    continue
                ZHratios.append(ZH)
                HTratios.append(HT)
                DPHIs.append(DPHI)
                AZs.append(AZ)

        # Assignment the ratios
        self.ZH, self.HT, self.DPHI, self.AZ = ZHratios, HTratios, DPHIs, AZs

        # Log. info
        startstr = self.starttime.strftime("%Y%m%d")
        endstr = self.endtime.strftime("%Y%m%d")
        logger.info(
            "Suc. Measuring {} during [{}-{}]".format(self.staid, startstr, endstr)
        )

    def criterions(
        self,
        phase_band=None,
        min_ratio=None,
        positive=None,
        maxzh=4,
        minzh=0,
        bin=1,
        max_iter_nb=3,
    ):
        """Perform two criterions to select the Rayleigh-like signals
        1.  Select signals whose phase shift between Z and R component
            is near 90 deg.
        2.  Select signals whose ratio of horizontal amplitude over the
            transport one is bigger than minimum ratio

        Parameters
        ==========
        phase_band: tuple
            Define the minimum and maximum acceptable phase shift for
            Rayleigh-like signal. Default value is None, meaning ignore
            the first criterion
        min_ratio: float
            the minimum H/T. Default value is None, meaning ignore
            the first criterion
        maxzh: float
            Define the maximum zh ratio bound
        minzh: float
            Define the minimum zh ratio bound
        bin: float
            bin for stacking
        positive: int
            -1 indicate range in (-maxPhase, -minPhase)
            1 indicate range in (minPhase, maxPhase)
            2 indicate range in (-maxPhase, -minPhase) and (minPhase, maxPhase)

        References
        ==========
        Tanimoto, Toshiro, Tomoko Yano, and Tomohiro Hakamata.
        "An approach to improve Rayleigh-wave ellipticity estimates from seismic
        noise: application to the Los Angeles Basin." Geophysical Journal International
        193.1 (2013): 407-420.
        """
        # Save constrain parameters
        self.phase_band, self.min_ratio = phase_band, min_ratio
        self.minzh, self.maxzh = minzh, maxzh
        self.positive = positive

        if phase_band:
            min_phase_shift, max_phase_shift = phase_band
            msk1ZH = self._phase_criterion(
                min_phase_shift=min_phase_shift, max_phase_shift=max_phase_shift
            )
        if min_ratio:
            msk2HT = self._HTratio_criterion(min_ratio=min_ratio)

        # Use two criterion to select signals
        RayZH, NonRayZH, SAVEDPHI, SAVEDAZ = [], [], [], []
        for tidx, item in enumerate(self.ZH):
            # loop over discretized frequency band
            Rayitem = np.zeros(item.shape, dtype=float)
            NonRayitem = np.zeros(item.shape, dtype=float)
            SAVEDPHIitem = np.zeros(item.shape, dtype=float)
            SAVEDAZitem = np.zeros(item.shape, dtype=float)
            for idx, value in enumerate(item):
                if msk1ZH[tidx][idx] and msk2HT[tidx][idx]:
                    if value > maxzh or value < minzh:
                        Rayitem[idx] = np.nan
                        SAVEDPHIitem[idx] = np.nan
                        SAVEDAZitem[idx] = np.nan
                    else:
                        Rayitem[idx] = value
                        SAVEDPHIitem[idx] = self.DPHI[tidx][idx]
                        SAVEDAZitem[idx] = self.AZ[tidx][idx]

                    NonRayitem[idx] = np.nan
                else:
                    Rayitem[idx] = np.nan
                    NonRayitem[idx] = value
                    SAVEDPHIitem[idx] = np.nan
                    SAVEDAZitem[idx] = np.nan
            RayZH.append(Rayitem), NonRayZH.append(NonRayitem)
            SAVEDPHI.append(SAVEDPHIitem), SAVEDAZ.append(SAVEDAZitem)
        # Assignment measurements
        self.RayZH, self.NonRayZH = RayZH, NonRayZH
        self.SAVEDPHI, self.SAVEDAZ = SAVEDPHI, SAVEDAZ
        # Get Median ZH ratios beneath this constrain condition
        meanZHs, stdZHs = self.get_ZHratio(bin=bin)

        torefine, iter_nb = True, 0
        while torefine:
            discard_nb = 0
            if iter_nb >= max_iter_nb:
                break
            # Discard ZH ratios those locate outside the 3*sigma area
            for tidx, item in enumerate(self.RayZH):
                # loop over discretized frequency band
                for idx, value in enumerate(item):
                    if np.abs(value - meanZHs[idx]) >= 3 * stdZHs[idx]:
                        self.RayZH[tidx][idx] = np.nan
                        self.SAVEDPHI[tidx][idx] = np.nan
                        self.SAVEDAZ[tidx][idx] = np.nan
                        discard_nb += 1
            meanZHs, stdZHs = self.get_ZHratio(bin=bin)
            # Jump out condition
            iter_nb += 1
            if discard_nb == 0:
                break

        # Get Median ZH ratios beneath this constrain condition
        self.meanZHs, self.stdZHs = self.get_ZHratio(bin=bin)
        fac = self.positive
        msg = "Suc. constrain measuements with dphi"
        logger.info(
            msg + " in {}*{} and HT >= {}".format(fac, self.phase_band, self.min_ratio)
        )

    def _phase_criterion(self, min_phase_shift=30, max_phase_shift=150):
        """Perform the first criterion, selecting signals where the
        phase shift between Z and R component is near 90 deg.

        Parameters
        ==========
        min_phase_shift: float
            the minimum acceptable phase shift for Rayleigh-like signal
        max_phase_shift: float
            the maximum acceptable phase shift for Rayleigh-like signal
        """

        def inner_phase(item, min_phase, max_phase):
            """return whether the value is enclosed in (min_phase, max_phase)
            or not
            """
            # phase_shift = np.nan_to_num(np.rad2deg(-1.0 * np.angle(item)))
            item = np.nan_to_num(item)
            msk = np.zeros_like(item)
            for idx, x in enumerate(item):
                if self.positive == -1.0:
                    msk[idx] = (x >= -1 * max_phase) * (x <= -1 * min_phase)
                elif self.positive == 1.0:
                    msk[idx] = (x <= max_phase) * (x >= min_phase)
                elif self.positive == 2.0:
                    tmp1 = (x >= -1 * max_phase) * (x <= -1 * min_phase)
                    tmp2 = (x <= max_phase) * (x >= min_phase)
                    msk[idx] = tmp1 + tmp2
            return msk

        msk1ZH = [
            inner_phase(item, min_phase_shift, max_phase_shift) for item in self.DPHI
        ]
        return msk1ZH

    def _HTratio_criterion(self, min_ratio=3):
        """Perform the second criterion, selecting signals where the
        wavforms is dominated by Rayleigh-like waves where the ratio
        of horizontal amplitude over transport amplitude is bigger
        than minimum ratio

        Parameters
        ==========
        min_ratio: float
            the minimum H/T
        """

        def judgement(x, min_rat):
            return np.nan_to_num(x) >= min_rat

        msk2HT = [judgement(item, min_ratio) for item in self.HT]
        return msk2HT

    def export(
        self,
        stalogfile=None,
        zhlogfile=None,
        htlogfile=None,
        phlogfile=None,
        azlogfile=None,
        only_stalog=False,
    ):
        """Export the measured ZH ratio, HTratio and statistical
        ZH ratios into log file

        Parameters
        ==========
        zhlogfile: str or path-like obj.
            LOG file path dirname to save zh ratios
        htlogfile: str or path-like obj.
            LOG file path dirname to save ht ratios
        stalogfile: str or path-like obj.
            LOG file path dirname to save mean and std of ZH ratios
        phlogfile: str or path-like obj.
            LOG file path dirname to save phase difference between
            horizontal and vertical components
        azlogfile: str or path-like obj.
            LOG file path dirname to save back-azimuth
        """

        def exfmt(x):
            return " ".join(["{:.5f}".format(item) for item in x]) + "\n"

        def time(y):
            return y.strftime("%Y%m%d")

        if stalogfile:
            try:
                selections = "# Phase band ({},{}), Mininum H/T ratio {}\n".format(
                    self.phase_band[0], self.phase_band[1], self.min_ratio
                )
                with open(stalogfile, "w") as f:
                    # export header
                    header = "# Statistical analysis of ZH Ratio of"
                    header += " {}@[{}-{}]\n".format(
                        self.staid, time(self.starttime), time(self.endtime)
                    )
                    header += selections
                    header += "Line one: period(s)\n"
                    header += "line two: Averaged ZH ratios at particular period\n"
                    header += "line three: STD ZH ratios at particular period\n"
                    header += (
                        "line faour: uncertainty of ZH ratios at particular period\n"
                    )
                    f.write(header)

                    # export data
                    f.write(exfmt(self.periods))
                    f.write(exfmt(self.meanZHs))
                    f.write(exfmt(self.stdZHs))
                if only_stalog:
                    return
            except TypeError as err:
                logger.error("NoDataError")

        selections = "# No Data Slection\n"
        if zhlogfile:
            with open(zhlogfile, "w") as f:
                # export header
                header = "# All ZH Ratio measurements of"
                header += " {}@[{}-{}]\n".format(
                    self.staid, time(self.starttime), time(self.endtime)
                )
                header += selections
                header += "Line one: period(s)\n"
                header += "Other lines: ZH ratios at particular period\n"
                f.write(header)
                # export data
                f.write(exfmt(self.periods))
                for item in self.ZH:
                    f.write(exfmt(item))

        if htlogfile:
            with open(htlogfile, "w") as f:
                # export header
                header = "# All HT Ratio measurements of"
                header += " {}@[{}-{}]\n".format(
                    self.staid, time(self.starttime), time(self.endtime)
                )
                header += selections
                header += "Line one: period(s)\n"
                header += "Other lines: HT ratios at particular period\n"
                f.write(header)

                # export data
                f.write(exfmt(self.periods))
                for item in self.HT:
                    f.write(exfmt(item))

        if phlogfile:
            with open(phlogfile, "w") as f:
                # export header
                header = "# Statistical analysis of ZH Ratio of"
                header += " {}@[{}-{}]\n".format(
                    self.staid, time(self.starttime), time(self.endtime)
                )
                header += selections
                header += "Line one: period(s)\n"
                header += "line two: phase difference between horizontal and"
                header += "vertical components at particular period\n"
                f.write(header)

                f.write(exfmt(self.periods))
                for item in self.DPHI:
                    f.write(exfmt(item))

        if azlogfile:
            with open(azlogfile, "w") as f:
                # export header
                header = "# Statistical analysis of ZH Ratio of"
                header += " {}@[{}-{}]\n".format(
                    self.staid, time(self.starttime), time(self.endtime)
                )
                header += selections
                header += "Line one: period(s)\n"
                header += "line two: Back-azimuth of rayleigh wave\n"
                f.write(header)

                f.write(exfmt(self.periods))
                for item in self.AZ:
                    f.write(exfmt(item))

    def plot_measurements(self, visual_period_band=None, prefix=None, refzh=None):
        """Plot the measurement for particular station during specific time period

        Parameters
        ==========
        visual_period_band: list or numpy array
            period band to visualize its measurements and criterions, default
            to be None which means visualize all freqeuency in self.periods. If given,
            the list or numpy array must be subset of self.periods.
        prefix: str
            prefix of all exported measurements figures. Defult to be ./self.staid
        refzh: dict.
            Give the reference ZH ratio
        """
        # Specific frequency band and prefix
        if not prefix:
            prefix = "./" + self.staid
        if not visual_period_band:
            visual_period_band = self.periods

        # Depict measurement for each frequency
        os.makedirs(prefix, exist_ok=True)
        exportfile = join(prefix, "{}.pdf".format(self.staid))

        with PdfPages(exportfile) as pdf:
            self._Measurement_per_period(pdf, startpage=True, refzh=refzh)
            for period in visual_period_band:
                periodidx = np.argmin(np.abs(period - self.periods))
                self._Measurement_per_period(pdf, periodidx=periodidx, startpage=False)
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d["Title"] = "ZH ratios"
            d["Author"] = "Xiao Xiao"
            d["Keywords"] = "ZH ratio measurements"
            d["CreationDate"] = datetime.datetime(2018, 10, 13)

    def _Measurement_per_period(self, pdf, periodidx=None, startpage=True, refzh=None):
        """Depict measurements for specific period

        Parameters
        ==========
        periodidx: int
            index of the specified period in self.periods
        pdf: PdfPages
            object of pdf which can depict on it
        startpage: bool
            judge if depict the final mean zh ratio or the
            judge process of each freqeuency
        refzh: dict.
            Give the reference ZH ratio
        """

        if startpage:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 6))
            # Comparison between synthetic and measurement graphyly
            ax.plot(self.periods, self.meanZHs, "g-")
            ax.errorbar(
                self.periods,
                self.meanZHs,
                marker="s",
                mec="red",
                yerr=self.stdZHs,
                label="Measured",
            )
            if refzh:
                ax.plot(refzh["period"], refzh["zh"], label="Synthetic")
            ax.set_xlim(self.periods.min(), self.periods.max())
            ax.set_xlabel("Period (s)")
            ax.set_ylabel("Saved ZH Ratio")
            ax.legend()

            # Save current figure
            pdf.savefig(fig)
            plt.close()
        else:
            # Obtain measurements
            def gfilter(x):
                return x[~np.isnan(x)]

            rayzhs = gfilter(np.array([item[periodidx] for item in self.RayZH]))
            hts = gfilter(np.array([item[periodidx] for item in self.HT]))
            dphis = gfilter(np.array([item[periodidx] for item in self.DPHI]))
            az = gfilter(np.array([item[periodidx] for item in self.SAVEDAZ]))

            # Visualization process
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))
            try:
                axes[0].hist(hts, bins=40, alpha=0.75, label="HT Ratios")
                axes[0].set_ylabel("Number")
                axes[0].set_title(
                    "HT ratios of all segments @{:.5f}s".format(self.periods[periodidx])
                )
                axes[0].axvline(x=self.min_ratio, color="r", label="HT limitation")
                axes[0].legend()
            except Exception as err:
                logger.info("Unhandled Error [{}]".format(err))

            # Combine all the phases
            # phases = np.rad2deg(-1.0 * np.angle(zhs))
            binnum = max(int((dphis.max() - dphis.min()) / 10), 1)
            axes[1].hist(
                dphis, bins=binnum, range=(-180, 180), alpha=0.75, label="Phase Shifts"
            )
            axes[1].set_ylabel("Number")
            axes[1].set_title(
                "Phase shifts of all segments @{:.5f}s".format(self.periods[periodidx])
            )
            fac = self.positive
            if self.positive == -1:
                axes[1].axvline(
                    x=-1 * self.phase_band[0], color="r", label="Phase Band"
                )
                axes[1].axvline(x=-1 * self.phase_band[1], color="r")
            elif self.positive == 1:
                axes[1].axvline(x=self.phase_band[0], color="r", label="Phase Band")
                axes[1].axvline(x=self.phase_band[1], color="r")
            elif self.positive == 2:
                axes[1].axvline(
                    x=-1 * self.phase_band[0], color="r", label="Phase Band"
                )
                axes[1].axvline(x=-1 * self.phase_band[1], color="r")
                axes[1].axvline(x=self.phase_band[0], color="r", label="Phase Band")
                axes[1].axvline(x=self.phase_band[1], color="r")

            axes[1].legend()

            # Depict all saved azimuth and azimuth stacking illustration if applied
            axes[2].scatter(np.rad2deg(az), rayzhs, alpha=0.5, label="Raw")
            axes[2].axhline(y=self.minzh, color="r", label="ZH ratio bound")
            axes[2].axhline(y=self.maxzh, color="r")
            axes[2].set_ylabel("ZH ratio")
            axes[2].set_xlabel("Baz (deg)")
            axes[2].set_title(
                "Back Azimuth of saved ZH ratios @{:.5f}s".format(
                    self.periods[periodidx]
                )
            )

            # Depict all saved ZH ratios
            if rayzhs.shape[0] != 0:
                binnum = max(int((rayzhs.max() - rayzhs.min()) / 0.1), 1)
                binnum = min(binnum, 40)
                axes[3].hist(rayzhs, bins=binnum, alpha=0.75, label="Raw")
                axes[3].set_ylabel("Number")
                if hasattr(self, "bins"):
                    msk = ~np.isnan(self.bin_means[:, periodidx])
                    bins = np.rad2deg(self.bins[msk])
                    bin_means = self.bin_means[:, periodidx][msk]

                    # Depict ZH ratio histogram
                    axes[3].twinx()
                    binnum = max(int((bin_means.max() - bin_means.min()) / 0.1), 1)
                    binnum = min(binnum, 40)
                    axes[3].hist(
                        bin_means,
                        bins=binnum,
                        alpha=0.75,
                        label="bin-stacked({:.2f})".format(bins[1] - bins[0]),
                    )
                    # Linear regression
                    model = LinearRegression(fit_intercept=True)
                    model.fit(bins.flatten()[:, np.newaxis], bin_means.flatten())
                    xfit = np.arange(0, 180, 5)
                    yfit = model.predict(xfit[:, np.newaxis])

                    # Depict distribution of bin-stacked zhs
                    axes[2].scatter(
                        bins, bin_means.flatten(), alpha=0.5, label="bin stacked"
                    )
                    axes[2].plot(
                        xfit, yfit, label="Regression({:.5f})".format(model.coef_[0])
                    )
                    axes[2].legend()

                mean_zh, std_zh = self.meanZHs[periodidx], self.stdZHs[periodidx]
                axes[3].axvline(x=mean_zh, color="r", label="Mean ZH ratio")
                axes[3].axvline(x=mean_zh - std_zh, color="r", label=r"$1 \sigma$")
                axes[3].axvline(x=mean_zh + std_zh, color="r")
                axes[3].legend()

                axes[3].set_ylabel("Number")
                axes[3].set_title(
                    "ZH ratios of Rayleigh-like segments @{:.5f}s".format(
                        self.periods[periodidx]
                    )
                )

            # Save current figure
            pdf.savefig(fig)
            plt.close()

    def get_ZHratio(self, bin=1):
        """Obtain mean ZH ratio and its uncertainty with bin-resampling
        method

        Parameters
        ==========
        bin: float
            azimuth bin in stacking
        """

        # Obtain all measurements and associated bazs
        def gfilter(x):
            return x[~np.isnan(x)]

        zhs = np.array([item for item in self.RayZH])
        bazs = np.array([item for item in self.SAVEDAZ])
        if bin is not None:
            # Assign bin group
            pi, radbin = np.pi, np.deg2rad(bin)
            bins = np.arange(0, pi, radbin)

            # stacking with azimuth bin
            bin_means = np.zeros((len(bins), len(self.periods))) * np.nan
            for period_idx in range(len(self.periods)):
                # Assign measurements into bin groups
                for bin_idx, value in enumerate(bins):
                    period_baz = bazs[:, period_idx]
                    msk = (period_baz >= value) * (period_baz <= value + radbin)
                    bin_means[bin_idx, period_idx] = np.nanmean(
                        zhs[:, period_idx][msk], axis=0
                    )
            self.bins, self.bin_means = bins, bin_means
            zhmean = np.nanmean(bin_means, axis=0)
            zhstd = np.nanstd(bin_means, axis=0)
        else:
            zhmean = np.nanmean(zhs, axis=0)
            zhstd = np.nanstd(zhs, axis=0)
        return zhmean, zhstd

    def import_measurements(
        self, zhlogfile, htlogfile, phlogfile, azlogfile, header=True
    ):
        """Same as the func. name

        Parameters
        ==========
        zhlogfile: str or path-like obj.
            log file of the ZH ratio measurements
        htlogfile: str or path-like obj.
            log file of the HT ratio measurements
        phlogfile: str or path-like obj.
            log file of the phase change for measurements
        header: bool
            determine the log file contains the header info. or not
        """

        # Check existence of class attribute
        def attrexistornot(classname, attrname):
            if not hasattr(classname, attrname):
                classname.__setattr__(attrname, [])

        attrexistornot(self, "ZH")
        attrexistornot(self, "HT")
        attrexistornot(self, "DPHI")
        attrexistornot(self, "AZ")

        # Import info. seperately
        period_zh, zh = self._import_value(zhlogfile, header=header)
        period_ht, ht = self._import_value(htlogfile, header=header)
        period_dphi, dphi = self._import_value(phlogfile, header=header)
        period_az, az = self._import_value(azlogfile, header=header)

        # Check if lose information
        if zh is None or ht is None or dphi is None or az is None:
            logger.error("LoseInfo [{}]".format(self.staid))
            return

        # check consistence of frequency band
        if not (
            period_zh.all() == period_ht.all() == period_dphi.all() == period_az.all()
        ):
            msg = "Period band mismatch [skipping {}]!".format(self.staid)
            logger.error(msg)
            return

        # Check the consistence between imported frequency band and
        # that stored in class
        if not hasattr(self, "periods"):
            self.periods = period_zh
        else:
            if not (self.periods == period_zh).all():
                msg = "Period band mismatch [skipping {}]!".format(self.staid)
                logger.error(msg)
                return

        # Check the content info. consistence
        if not (len(zh) == len(ht) == len(dphi) == len(az)):
            msg = "FileContentInconsistence [skipping {}]!".format(self.staid)
            logger.error(msg)

        # Append imported value to class dataset
        for idx in range(len(zh)):
            self.ZH.append(zh[idx])
            self.HT.append(ht[idx])
            self.DPHI.append(dphi[idx])
            self.AZ.append(az[idx])

    def _import_value(self, logfile, header=True):
        """Import zh/ht ratio or dphi from log file"""
        # set if the log file include the ratio or not
        if header:
            startidx = 4
        else:
            startidx = 0

        # File existence check
        try:
            with open(logfile) as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error("NoFileError [skiping {}]".format(logfile))
            return None, None

        # File content check
        if len(lines) <= startidx:
            logger.error("NoContentError [skiping {}]".format(logfile))
            return None, None

        # Resolve the file content and get the freqeuency
        periods = np.fromstring(lines[startidx].strip(), dtype=float, sep=" ")

        # Get later ZH/HT ratios or dphi
        values = []
        for idx in range(startidx + 1, len(lines)):
            subval = np.array([float(x) for x in lines[idx].strip().split(" ")])
            values.append(subval)
        return periods, values


class NoiseSlice(object):
    """Class to handle seismic noise during a time slice"""

    def __init__(self, starttime, endtime):
        """Define the Noise records

        Parameter
        =========
        starttime: ObsPy.UTCDateTime
            start time of this slice of seismic records
        endtime: ObsPy.UTCDateTime
            end time of this slice of seismic records
        """
        # Get starttime and endtime
        self.starttime = starttime
        self.endtime = endtime

    def Measure_ZH(
        self,
        rawst,
        periods,
        freq_grid,
        event_onset=6,
        event_offset=4.5,
        earthquake_minfreq=0.1,
        earthquake_maxfreq=0.3,
        treshold=10,
        gap_treshold=500,
        short_time=3,
        long_time=50,
        data_fill=0.95,
        debug=False,
    ):
        """Measure ZH ratio during this time slice
        Parameters
        ==========
        rawst: Class obspy.Stream
            raw waveform
        periods: numpy.array
            discrited period band
        freq_grid: float
            frequency band width
        onset: float
            first treshold for detecting earthquake
        offset: float
            second treshold for detecting earthquake
        earthquake_minfreq: float
            minimal frequency for filtering waveform
            in earthquake detection
        earthquake_maxfreq: float
            maximum frequency for filtering waveform
            in earthquake detection
        treshold: float
            the maximum energy ratio of E N component.
            For detecting instrumental response
            constant mismatch
        """
        # Trim slices waveforms
        st = deepcopy(rawst)
        st.trim(starttime=self.starttime, endtime=self.endtime)

        # Check data
        if len(st) != 3:
            raise ValueError("LoseComponent")

        # st.merge(fill_value='interpolate')
        # If data masked, ignore it
        # for tr in st:
        #    # Check portion of interplated data
        #    if (tr.data == 1.0e20).any():
        #        raise ValueError("LoseData")
        #    # Check portion of zero data
        #    msk = tr.data == 0
        #    if (len(tr.data[msk]) / tr.stats.npts) > 0.1:
        #        raise ValueError("LoseData")

        # Data filter
        grid = freq_grid
        # Decompose stream
        try:
            N, E, Z = seperate_channels(st, comps=["N", "E", "Z"])
        except IndexError:
            N, E, Z = seperate_channels(st, comps=["1", "2", "Z"])
        except Exception as err:
            raise ValueError("Unkown component")
        # Preprocess the raw seismic data, including Detrend, Taper and Filter
        for tr in [N, E, Z]:
            tr.detrend("simple")
            tr.taper(max_percentage=0.05)
            tr.filter(
                "bandpass",
                freqmin=(1.0 / periods.max() - grid),
                freqmax=(1.0 / periods.min() + grid),
                zerophase=True,
            )

        # Event detection and elimination
        event_st = deepcopy(rawst)
        event_st.trim(
            starttime=self.starttime - long_time,
            endtime=self.endtime,
        )
        event_st.merge(fill_value="interpolate")

        # preprocess
        event_st.detrend("simple")
        event_st.taper(max_percentage=0.05)
        event_st.filter(
            "bandpass",
            freqmin=earthquake_minfreq,
            freqmax=earthquake_maxfreq,
            corners=4,
        )
        # event_st[0].write("test.sac")
        # Decompose stream
        EN, EE, EZ = seperate_channels(event_st, comps=["N", "E", "Z"])

        # Classify waveform as noise or event by
        # traditional STA/LTA method with N traces
        df = EN.stats.sampling_rate
        cft_Z = classic_sta_lta(EZ.data, int(short_time * df), int(long_time * df))
        cft_E = classic_sta_lta(EE.data, int(short_time * df), int(long_time * df))
        cft_N = classic_sta_lta(EN.data, int(short_time * df), int(long_time * df))

        # Only consider middle one patch waveform
        msk_Z = np.arange(0, len(cft_Z)) >= int(long_time * df)
        gap_cft_z = 1.0 / (cft_Z[msk_Z] + MINVALUE)
        gap_len = len(gap_cft_z[gap_cft_z >= gap_treshold])
        gap_percentage = gap_len / EZ.stats.npts
        if gap_percentage >= (1 - data_fill):
            logger.debug("WithGap [{}]".format(self.starttime))
            raise ValueError("WithGap")

        # Merge three trigger machine
        msk_N = np.arange(0, len(cft_N)) >= int(long_time * df)
        msk_E = np.arange(0, len(cft_E)) >= int(long_time * df)
        trigger_N = trigger_onset(cft_N[msk_N], event_onset, event_offset)
        trigger_E = trigger_onset(cft_E[msk_E], event_onset, event_offset)
        trigger_Z = trigger_onset(cft_Z[msk_Z], event_onset, event_offset)
        triggers = np.concatenate((trigger_N, trigger_E, trigger_Z))
        if len(triggers) != 0:
            #  Merge time periods to eliminate
            logger.debug("WithEvent[{}]".format(self.starttime))
            raise ValueError("WithGap")

        # Check average energy ratio of N and E component, ignore those
        # higher than 10
        Ratio_EN = E.data.std() / N.data.std()
        if Ratio_EN >= treshold or Ratio_EN <= (1.0 / treshold):
            raise ValueError("E,N component energy mismatch !")

        # Save image of waveform
        if debug:
            filename = "./{}_{}.png".format(self.starttime, self.endtime)
            plt.plot(EZ.data, label="Raw")
            plt.plot(Z.data, label="Handled")
            plt.legend()
            plt.savefig(filename)
            plt.close()

        def Getspectrum(iptsignal, dt):
            """give out the freqeuency band and spectrum of  particular trace

            Parameter
            =========
            iptsignal: numpy.array
                seisic waveform
            """
            freq = fftfreq(iptsignal.size, d=dt)
            return freq, fft(iptsignal)

        # Perform FFT and get the spectrum of each waveforms
        deltat = 1.0 / Z.stats.sampling_rate
        freq, specZ = Getspectrum(Z.data, dt=deltat)
        _, specN = Getspectrum(N.data, dt=deltat)
        _, specE = Getspectrum(E.data, dt=deltat)

        # Get the rotated horizontal component
        ZH = np.zeros(periods.shape, dtype=float)
        DPHI = np.zeros(periods.shape, dtype=float)
        HT = np.zeros(periods.shape, dtype=float)
        AZ = np.zeros(periods.shape, dtype=float)
        for idx, item in enumerate(periods):
            # Search for signal azimuth
            lft_freq, right_freq = (1.0 / item) - grid, (1.0 / item) + grid
            ang, max_H, min_H, msk = Quan_I(freq, specN, specE, lft_freq, right_freq)
            # Get Z component spectrum at nearest frequency band
            AZ[idx] = ang

            # Estimate HT amplitude ratio
            # HT[idx] = np.abs(max_H).mean() / np.abs(min_H).mean()
            # Search for discretized frequency nearest to the target frequency
            near_f_idx = np.abs(freq[msk] - 1.0 / item).argmin()
            Zitem = specZ[msk][near_f_idx]
            Nitem = specN[msk][near_f_idx]
            Eitem = specE[msk][near_f_idx]
            Hitem = max_H[near_f_idx]
            Titem = min_H[near_f_idx]

            # Estimate ZH ratio
            # ZH[idx] = np.abs(Zitem) / np.abs(Hitem)

            # Estimate ZH ratio amplitude part based on average ratio
            # at nearby frequency narrow band
            # The Same as tanimoto 06
            ZH[idx] = np.abs(specZ[msk]).mean() / np.abs(max_H).mean()

            # Estimate ZH phase shift
            # DPHI[idx] = np.rad2deg(np.angle(Hitem) - np.angle(Zitem))
            upper = np.imag(Nitem) * np.cos(ang) + np.imag(Eitem) * np.sin(ang)
            lower = np.real(Nitem) * np.cos(ang) + np.real(Eitem) * np.sin(ang)
            PHAH = np.arctan2(upper, lower)
            DPHI[idx] = np.rad2deg(PHAH - np.angle(Zitem))

            # Estimate HT ratio
            HT[idx] = np.abs(Hitem) / np.abs(Titem)

            if debug:
                if (
                    HT[idx] > 3
                    and DPHI[idx] >= 60
                    and DPHI[idx] <= 120
                    and ZH[idx] < 0.5
                ):
                    # print("Z:", np.abs(Zinter).mean(), "H:", np.abs(max_H).mean(), "ZH:", ZH[idx])
                    print("AZ:", AZ[idx], "T:", np.abs(min_H).mean(), "HT:", HT[idx])
                    # plt.plot(E.filter("bandpass", freqmin=0.1, freqmax=0.3, zerophase=True).data)
                    # plt.plot(N.filter("bandpass", freqmin=0.1, freqmax=0.3, zerophase=True).data)
                    # plt.plot(Z.filter("bandpass", freqmin=0.1, freqmax=0.3, zerophase=True).data)
                    plt.plot(E.data)
                    plt.plot(N.data)
                    plt.plot(Z.data)
                    plt.show()

                    # st.plot()
                    # event_st.plot()
        return ZH, HT, DPHI, AZ


def sub_func(
    subslice,
    starttimes=None,
    endtimes=None,
    rawst=None,
    periods=None,
    freq_grid=None,
    **kwargs
):
    """wrapper for NoiseSlice Measure_ZH function

    Parameters
    ==========
    subslice: NoiseSlice obj.
        slice to compute its ZH ratio over all
        frequency band
    rawst: obspy.Stream class
        raw seismic continuous waveform
    periods: numpy.array
        discretized period points
    freq_grid:
        frequency band width for single frequency measurement
    """
    try:
        # remove segments with end time larger than starttime of the subslices
        subst = Stream()
        for idx, x in enumerate(endtimes):
            if subslice.starttime > x or subslice.endtime < starttimes[idx]:
                continue
            else:
                subst.append(rawst[idx])

        ZH, HT, DPHI, AZ = subslice.Measure_ZH(
            rawst=subst, periods=periods, freq_grid=freq_grid, **kwargs
        )
    except Exception as err:
        logger.debug("Unhandled Error [{}]".format(err))
        ZH, HT, DPHI, AZ = None, None, None, None
    return ZH, HT, DPHI, AZ


def sub_slice(idx, gaps=None, timepoints=None, step=None, timelen=None):
    """Judge if the time period"""
    startt, endt = timepoints[idx], timepoints[idx + step]

    # Check overlap of target time period and the gaps
    overlaps = [x & P.closed(startt, endt) for x in gaps]
    overlap_len = (
        np.array([x.upper - x.lower for x in overlaps if x.lower != P.inf]) / timelen
    )
    if overlap_len.sum() >= 0.5:
        return None
    else:
        return NoiseSlice(startt, endt)
