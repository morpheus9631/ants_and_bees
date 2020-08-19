# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.ROOT_PATH = "D:\\GitWork\\ants_and_bees\\"
# _C.WORK.ROOT_PATH = "/home/user/work/ants_and_bees/"

_C.DATA = CN()
_C.DATA.ROOT_PATH = "D:\\GitWork\\ants_and_bees\\data\\"
# _C.DATA.ROOT_PATH = "/home/user/work/ants_and_bees/data/"
_C.DATA.TRAIN_DIR = "train"
_C.DATA.VALIDATE_DIR = "val"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()