from yacs.config import CfgNode as CN
from .model_path import MODEL_PATH

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "bninception"
#_C.MODEL.BACKBONE.NAME = "resnet50"

_C.MODEL.PRETRAIN = 'imagenet'
_C.MODEL.PRETRIANED_PATH = MODEL_PATH

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = "linear_norm"
_C.MODEL.HEAD.DIM = 512

_C.MODEL.WEIGHT = ""

# Checkpoint save dir
_C.SAVE_DIR = '/home/songkun/PycharmProjects/TPAMI/script/output'

# Loss
_C.LOSSES = CN()
_C.LOSSES.NAME = 'anml_loss'

# ANML loss
_C.LOSSES.ANML_LOSS = CN()
_C.LOSSES.ANML_LOSS.SCALE_POS = 1.53
_C.LOSSES.ANML_LOSS.SCALE_NEG = 30
_C.LOSSES.ANML_LOSS.HARD_MINING = True


# Data option
_C.DATA = CN()
_C.DATA.TRAIN_IMG_SOURCE = '/data/songkun/CUB_200_2011/train.txt'
_C.DATA.TEST_IMG_SOURCE = '/data/songkun/CUB_200_2011/test.txt'
_C.DATA.TRAIN_BATCHSIZE = 70
_C.DATA.TEST_BATCHSIZE = 256
_C.DATA.NUM_WORKERS = 8
_C.DATA.NUM_INSTANCES = 9

# Input option
_C.INPUT = CN()

# INPUT CONFIG
_C.INPUT.MODE = 'BGR'
_C.INPUT.PIXEL_MEAN = [104. / 255, 117. / 255, 128. / 255]
_C.INPUT.PIXEL_STD = 3 * [1. / 255]

_C.INPUT.FLIP_PROB = 0.5
_C.INPUT.ORIGIN_SIZE = 256
_C.INPUT.CROP_SCALE = [0.16, 1]
_C.INPUT.CROP_SIZE = 227

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.IS_FINETURN = False
_C.SOLVER.FINETURN_MODE_PATH = ''
_C.SOLVER.MAX_ITERS = 4000
_C.SOLVER.STEPS = [1000, 2000, 3000]
_C.SOLVER.OPTIMIZER_NAME = 'SGD'
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.BIAS_LR_FACTOR = 1
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 200
_C.SOLVER.WARMUP_METHOD = 'linear'
_C.SOLVER.CHECKPOINT_PERIOD = 200
_C.SOLVER.RNG_SEED = 1

# Logger
_C.LOGGER = CN()
_C.LOGGER.LEVEL = 20
_C.LOGGER.STREAM = 'stdout'

# Validation
_C.VALIDATION = CN()
_C.VALIDATION.VERBOSE = 200
_C.VALIDATION.IS_VALIDATION = True