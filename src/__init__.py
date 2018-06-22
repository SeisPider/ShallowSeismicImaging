"""polarPy
A developing software to image surface seismic wave velocity
"""
import doctest
import logging

# Setup the logger
FORMAT = "[%(asctime)s]  %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# set earth's radius
EARTH_R = 6371

__title__ = "polarPy"
__version__ = "0.0.1"
__author__ = "Xiao Xiao"
__license__ = "MIT"
__copyright__ = "Copyright 2017-2018 Xiao Xiao"

