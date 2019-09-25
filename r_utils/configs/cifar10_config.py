from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "CIFAR10"
__C.DOWNLOAD = True


# Network parameters
#
__C.NN = edict()
__C.NN.REGIME = "REGULAR"

# for image color scale
__C.NN.COLOR = 3
# 32x32 images resizied to 128x128
__C.NN.IMG_SIZE = 32

__C.NN.NUM_CLASSES = 10
