import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTSPEECH_DIR = os.path.join(BASE_DIR, "data_ArtSpeech", "training")

# String constants
TRAIN = "train"
VALID = "validation"
TEST = "test"

LOWER_LIP = "lower-lip"
SOFT_PALATE = "soft-palate"
TONGUE = "tongue"
UPPER_LIP = "upper-lip"

GRAPH_BASED = "graph-based"
ACTIVE_CONTOURS = "active-contours"
SKELETON = "skeletonize"
