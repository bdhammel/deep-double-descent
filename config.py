from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = 'cuda:0'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.LEARNING_RATE = 1e-3
_C.TRAIN.STEPS = 2_000_000

_C.MODEL = CN()
_C.MODEL.D = 3
_C.MODEL.W = 2

_C.DATA = CN()
_C.DATA.N_POINTS = 1000
_C.DATA.NOISE = .5

cfg = _C
