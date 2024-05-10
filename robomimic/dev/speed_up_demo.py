import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase


demo_fn = "/home/nadun/Data/phd_project/robomimic/datasets/lift/ph/all_obs_v141.hdf5"