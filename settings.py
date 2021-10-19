import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTSPEECH_DIR = os.path.join(BASE_DIR, "data_ArtSpeech", "training")

# String constants
TRAIN = "train"
VALID = "validation"
TEST = "test"

ARYTENOID_MUSCLE = "arytenoid-muscle"
EPIGLOTTIS = "epiglottis"
LOWER_INCISOR = "lower-incisor"
LOWER_LIP = "lower-lip"
PHARYNX = "pharynx"
SOFT_PALATE = "soft-palate"
THYROID_CARTILAGE = "thyroid-cartilage"
TONGUE = "tongue"
UPPER_INCISOR = "upper-incisor"
UPPER_LIP = "upper-lip"
VOCAL_FOLDS = "vocal-folds"

GRAPH_BASED = "graph-based"
ACTIVE_CONTOURS = "active-contours"
SKELETON = "skeletonize"
