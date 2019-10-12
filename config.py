"""
Constants and paths used in the methods for the dataset generation
"""
from utils.utils import *

"""
Constants for bin extraction and dataset generation
"""

# coordinates of the Touolouse map from the source file
GLOBAL_X_MIN = 1.24951
GLOBAL_Y_MIN = 43.4682
GLOBAL_X_MAX = 1.695996
GLOBAL_Y_MAX = 43.744981

# width and height of the map
GLOBAL_DX = GLOBAL_X_MAX - GLOBAL_X_MIN
GLOBAL_DY = GLOBAL_Y_MAX - GLOBAL_Y_MIN

# step representing the side of the square in the bins and datapoints to be generated
STEP = 0.001

# number of bins using the current step
N_X_BINS = int(GLOBAL_DX / STEP)  # 223
N_Y_BINS = int(GLOBAL_DY / STEP)  # 138
N_TOTAL_BINS = N_X_BINS * N_Y_BINS  # 30774

"""
Constants for dataset generation
"""

# constraints on the number of nodes and edges for graph filtering when generating the dataset
MIN_NUM_NODES = 5
MAX_NUM_NODES = 9
MIN_NUM_EDGES = 4
MAX_NUM_EDGES = 15

DISTANCE_THRESHOLD = 0.0002  # maximum distance for nodes to be considered lying in the same point (merging)
ALPHA_THRESHOLD = 15  # maximum angle between two consecutive segments to be approximated with a unique, straight edge

# Expressing the map coordinates and width height using the Point class
GLOBAL_MIN = Point(1.24951, 43.4682)
GLOBAL_MAX = Point(1.695996, 43.744981)
GLOBAL_D = Point(GLOBAL_DX, GLOBAL_DY)

# Number of possible datapoints resulting from translation with augmentation
N_POSSIBLE_DATAPOINTS = N_TOTAL_BINS * 4 ** 2  # 492384

# coordinates as fractions for the centers for validation and test regions in the map, for the respective splits
VALID_SPLIT_CENTER = (.6, .7)
TEST_SPLIT_CENTER = (.3, .4)
WIDTH_SPLIT_FRACTION = .025  # half of the width/height for the test/valid region

# Unit square using the Square object
UNIT_SQUARE = Square(1., -1., -1., 1.)

#width and height of valid and test regions using coordinates
step_region = {
    "x": WIDTH_SPLIT_FRACTION * GLOBAL_DX,
    "y": WIDTH_SPLIT_FRACTION * GLOBAL_DY
}

# coordinates of valid region center
valid_region = {
    "x": GLOBAL_X_MIN + VALID_SPLIT_CENTER[0] * GLOBAL_DX,
    "y": GLOBAL_Y_MIN + VALID_SPLIT_CENTER[1] * GLOBAL_DY
}

# coordinates of test region center
test_region = {
    "x": GLOBAL_X_MIN + TEST_SPLIT_CENTER[0] * GLOBAL_DX,
    "y": GLOBAL_Y_MIN + TEST_SPLIT_CENTER[1] * GLOBAL_DY
}

SPLIT_NAMES = {0: 'train', 1: 'valid', 2: 'test', 3: 'augment'}

"""
PATHS for dataset generation
"""

PATH_DATASET = "dataset/{}/".format(STEP)
PATH_IMAGES = {
    0: PATH_DATASET + "{}/images/".format(SPLIT_NAMES[0]),
    1: PATH_DATASET + "{}/images/".format(SPLIT_NAMES[1]),
    2: PATH_DATASET + "{}/images/".format(SPLIT_NAMES[2]),
    3: PATH_DATASET + "{}/images/".format(SPLIT_NAMES[3]),
}

PATH_FILES = {
    0: PATH_DATASET + "{}.pickle".format(SPLIT_NAMES[0]),
    1: PATH_DATASET + "{}.pickle".format(SPLIT_NAMES[1]),
    2: PATH_DATASET + "{}.pickle".format(SPLIT_NAMES[2]),
    3: PATH_DATASET + "{}.pickle".format(SPLIT_NAMES[3]),
}

# check for existence of the directories, otherwise create them
ensure_dir(PATH_DATASET)
for d in PATH_IMAGES.values():
    ensure_dir(d)
    ensure_dir(d + "/extra_plots")

