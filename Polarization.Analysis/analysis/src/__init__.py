# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
#      Purpose: Modulus
#       Status: Developing
#   Dependence: Python 3.6
#      Version: ALPHA
# Created Date: 10:51h, 13/05/2018
#       Author: Xiao Xiao, https://github.com/SeisPider
#        Email: xiaox.seis@gmail.com
#     Copyright (C) 2017-2017 Xiao Xiao
#-------------------------------------------------------------------------------
"""polarPy
A developing software to image surface seismic wave velocity by measuring 
apparent polarization angle of particular phase. 
"""
import doctest
import logging

# Setup the logger
FORMAT = "[%(asctime)s]  %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
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

