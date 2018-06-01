#! /usr/bin/env python
import numpy as np
from obspy import UTCDateTime
from os.path import join, basename
from glob import glob
import logging
import datetime as dt
# Setup the logger
FORMAT = "[%(asctime)s]  %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SourceResponse(object):
    """class to handle all response files
    """

    def __init__(self, subdir="./"):
        """initialize and import response files

        Parameter
        =========
        subdir : str or path-like obj.
            subdir of response files
        """
        # initialize files location
        self.subdir = subdir

        # set source code
        self.source = "CENC"  # CENC

        # scan stations
        self.response_scanner(suffix="SACPZs")

    def __repr__(self):
        """representation
        """
        return "<Response files of {}>".format(self.source)

    def response_scanner(self, suffix="SACPZs"):
        """scan response files 
        """
        # scan networks names
        networkfolders = glob(join(self.subdir, "_".join(["*", suffix])))

        def obtain_networks(folders):
            """obtain networks' name from glob list

            Parameter
            =========
            folders : list
                folders of networks
            """
            return [(basename(folder).split("_")[0], folder) for folder in folders]

        # obtain all networks
        networks = obtain_networks(networkfolders)

        self.response = {}
        # obtain all network responses
        for network in networks:
            name, folder = network
            self.response.update({name: NetworkResponse(network)})
    
    def response_files_extractor(self, time):
        """Return location of response files at particular time

        Parameter
        =========
        time : `~ObsPy.UTCDateTime`
            time to extract the response files
        """
        network_responses = []
        for key, value in self.response.items():
            print(key, value)
            network_responses.append(value.loop_for_event(time))
        return network_responses

class NetworkResponse(object):
    """class to handle response file of an entire network
    """

    def __init__(self, network, prefix="PZs"):
        """initialization
        """
        self.network = network
        self.prefix = prefix
        self.responses = self.import_responsefiles()

    def __repr__(self):
        """representation
        """
        return "<Response files of network {}>".format(self.network[0])

    def loop_for_event(self, time):
        """Return response files at particular time
        
        Parameter
        =========
        time : `~ObsPy.UTCDateTime`
        """
        response = {}
        for key, value in self.responses.items():
            response.update({key:value.get_response(time)})
        return response
        
    def import_responsefiles(self):
        """Initialization of a network response
        """
        # obtain files list
        name, folder = self.network
        responsefiles = glob(join(folder, "_".join([self.prefix, name, "*"])))

        def time_checker(timestr):
            """Check info. included in time string

            Parameter
            =========
            timestr : str
                time info
            """

            # Check string type : YYYYMM or YYYYMMDD
            # If YYYYMM, set it to be first day this month
            if len(timestr) == 6:
                return UTCDateTime(timestr + "01")
            elif len(timestr) == 8:
                return UTCDateTime(timestr)
            else:
                logger.error("Can't resolve time string")
                return None

        # obtain response files
        response = {}
        for respfile in responsefiles:
            spliter = basename(respfile).split("_")
            if len(spliter) == 6:
                _, net, sta, cha, startt, endt = spliter
                starttime, endtime = time_checker(startt), time_checker(endt)
                # starttime equals endtime, indicating this file only works
                # during this month
                if starttime == endtime:
                    endtime += dt.timedelta(days=30)

            elif len(spliter) == 5:
                _, net, sta, cha, startt = spliter
                endt = UTCDateTime(dt.datetime.now())
                starttime, endtime = time_checker(startt), endt

            # channel id
            trid = ".".join([net, sta, "00", cha])
            if trid not in response.keys():
                response.update({trid: TraceResponse(trid)})

            if not starttime or not endtime:
                logger.error("NoTimeInfo {}".format(".".join([net, sta, cha])))
                continue
            else:
                response[trid].update_periods(starttime, endtime, respfile)
        return response


class TraceResponse(object):
    """class to handle response file of a particular trace
    """

    def __init__(self, trid):
        """response class initialization of this trace
        """
        self.trace = trid
        self.periods = []

    def __repr__(self):
        """representation
        """
        return "Response file of trace {}".format(self.trace)

    def update_periods(self, starttime, endtime, filedirname):
        """obtain periods and response file of this period

        Parameter
        =========
        starttime : `~obspy.UTCDateTime`
            starttime of this period
        endtime : `~obspy.UTCDateTime`
            endtime of this period
        filedirname : str or path-like obj.
            response file dirname of at this period
        """
        self.periods.append((TimePeriod(starttime, endtime), filedirname))

    def get_response(self, time):
        """Return location of response file base on inputted time

        Parameter
        =========
        time : `~obspy.UTCDateTime`
           time to obtain response file 
        """
        timediffs = np.zeros(len(self.periods))
        for index, period in enumerate(self.periods):
            time_period, filename = period
            if time_period.includeornot(time):
                return filename
            else:
                timediffs[index] = time_period.obtain_timediff(time)

        # find a nearest time
        indexmin = timediffs.argmin()

        # if no file is finded
        logger.info("Choose Nearest time period for {}".format(time))
        return self.periods[indexmin][1]


class TimePeriod(object):
    """class to indicate a time period
    """

    def __init__(self, starttime, endtime):
        """initilization

        Parameter
        =========
        starttime : `~obspy.UTCDateTime`
            starttime of this period
        endtime : `~obspy.UTCDateTime`
            endtime of this period
        """
        self.starttime = starttime
        self.endtime = endtime

    def __repr__(self):
        """representation
        """
        return "<Time period {}-{}>".format(self.strftime("%Y%m%d"),
                                            self.strftime("%Y%m%d"))

    def includeornot(self, time):
        """To judge if time in this time period

        Parameter
        =========
        time : `~obspy.UTCDateTime`
           time to judge 
        """
        if time >= self.starttime and time <= self.endtime:
            return True
        else:
            return False

    def obtain_timediff(self, time):
        """calculate difference between time and time period

        Parameter
        =========
        time : `~obspy.UTCDateTime`
           time to calculate 
        """
        return min(abs(time - self.starttime), abs(time - self.endtime))


if __name__ == "__main__":
    sourceresponse = SourceResponse(subdir="./Response")
    #AH = sourceresponse.response['XJ']
    #trresp = AH.responses['XJ.AKS.00.BHZ']
    #print(trresp.get_response(UTCDateTime("20170501")))
    responses = sourceresponse.response_files_extractor(UTCDateTime("20170501"))
    for i in responses:
        print(i)
