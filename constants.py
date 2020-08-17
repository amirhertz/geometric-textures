import os

EPSILON = 1e-6
DEBUG = False
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/dataset'
RAW_MESHES = f'{DATA_ROOT}/raw'
CACHE_ROOT = f'{DATA_ROOT}/cache'
CHECKPOINTS_ROOT = f'{PROJECT_ROOT}/checkpoints'
