import sys
sys.dont_write_bytecode = True

import os
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import core
import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args