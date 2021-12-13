import os
import sys
import inspect
import urllib.request
from datetime import datetime
import shutil

import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


TRAIN_RATE = 0.8
SEQ_LEN = 7
PRE_LEN = 2

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d, %H:%M:%S'


def main():


	


if __name__ == '__main__':
	main()