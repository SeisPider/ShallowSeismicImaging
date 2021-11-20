from glob import glob
import numpy as np
from os.path import join
import os

if __name__ == "__main__":
    eventdirs = glob("../../IRIS/P.waves/20*")
    for eventdir in eventdirs:
        print(eventdir)
        eventid = eventdir.split("/")[-1]

        filenames = glob(join(eventdir, "YP.NE73*"))
        if len(filenames) == 0:
            continue
        else:
            exdir = "./{}".format(eventid)
            os.makedirs(exdir)
            for filename in filenames:
                os.system("cp {} {}".format(filename, exdir))
