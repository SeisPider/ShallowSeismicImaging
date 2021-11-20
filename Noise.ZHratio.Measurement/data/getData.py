from glob import glob
import numpy as np
from os.path import join
import os

if __name__ == "__main__":
    datedirs = glob("../../../../../CHINA/Instrumental_Removed/IRIS/201*")
    for datedir in datedirs:
        print(datedir)
        date = datedir.split("/")[-1]

        filenames = glob(join(datedir, "YP.NE73*"))
        if len(filenames) == 0:
            continue
        else:
            exdir = "./{}".format(date)
            os.makedirs(exdir)
            for filename in filenames:
                os.system("cp {} {}".format(filename, exdir))
