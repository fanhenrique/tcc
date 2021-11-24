import inspect
import os
import logging

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

PATH = path+'/out'
PATH_FIGS = PATH+'/figs-graphs'
PATH_GRAPHS = PATH+'/out-graphs'
PATH_MATRICES = PATH+'/out-matrices' 

TRACKER = 'TRACKER'
MONITOR = 'MONITOR'
PEER = 'PEER'
MASTERSERVER = 'MASTER SERVER'

COLOR_TRACKER = 'red'
COLOR_MONITOR = 'blue'
COLOR_PEER = 'green'
COLOR_MASTERSERVER = 'yellow'

SHOWMASTER = None
SHOWPEERS = None
SHOWTRACKERS = None
SHOWMONITORS = None

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'
