# config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.WORK = CN()
_C.WORK.ROOT_PATH = "D:\\GitWork\\ants_and_bees\\"
# _C.WORK.ROOT_PATH = "/home/user/work/ants_and_bees/"

_C.DATA = CN()
_C.DATA.ROOT_PATH = "D:\\GitWork\\ants_and_bees\\data\\hymenoptera_data\\"
# _C.DATA.ROOT_PATH = "/home/user/work/ants_and_bees/data/hymenoptera_data/"


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()