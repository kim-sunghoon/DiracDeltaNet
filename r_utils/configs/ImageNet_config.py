from easydict import EasyDict as edict
import os
__C = edict()
cfg = __C

# Data set config

__C.DATASET_NAME = "ImageNet"
__C.DOWNLOAD = True

#
# Network parameters
#
__C.NN = edict()
__C.NN.REGIME = "SMALL"

# for image color scale
__C.NN.COLOR = 3
# Resizied to 224x224 in torchvision.transforms.Resize()
__C.NN.IMG_SIZE = 224
__C.NN.NUM_CLASSES = 1000
