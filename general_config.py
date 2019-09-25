import os
from easydict import EasyDict as edict
import time

__C = edict()
cfg = __C

# set your dataset directory
__C.DATASET_DIR = "/home/shkim/dataset"

# default dataset directory
if not os.path.isdir(__C.DATASET_DIR):
    __C.DATASET_DIR = "./dataset/"
    if not os.path.isdir(__C.DATASET_DIR):
        os.mkdir(__C.DATASET_DIR)


# Debug parameters
__C.PRINT_FREQ = 10
__C.SAVE_FREQ = 1000
__C.VAL_FREQ = 3
# For reproducibility
__C.RND_SEED = 3

### Enable Visdom for loss visualization
### install: pip install visdom
### execute: python -m visdom.server
### access:  http://localhost:8097
#  __C.VISDOM = True
#
#  if cfg.VISDOM:
#      from visdom import Visdom
#      __C.vis = Visdom()
#      __C.loss_window = ""

