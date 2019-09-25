from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "MNIST"
__C.DOWNLOAD = True


# Network parameters
#
__C.NN = edict()
__C.NN.REGIME = "REGULAR"

# for image color scale: gray scale
__C.NN.COLOR = 1
# MNIST images are 28x28, but resizied to 64x64

__C.NN.IMG_SIZE = 28
__C.NN.NUM_CLASSES = 10


