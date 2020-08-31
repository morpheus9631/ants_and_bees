# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.ROOT_PATH = ""

_C.DATA = CN()
_C.DATA.RAW_PATH = ""
_C.DATA.PROCESSED_PATH = ""

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.NUM_EPOCHS = 25
_C.TRAIN.LEARNING_RATE = 0.001
_C.TRAIN.MOMENTUM = 0.9


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()