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

# Post-processing configuration per class
POST_PROCESSING = {
    LOWER_LIP: dict(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        threshold = 0.3
    ),
    SOFT_PALATE: dict(
        method=ACTIVE_CONTOURS,
        alpha = 0.00015,
        beta=1000,
        gamma=0.1,
        threshold=0.4
    ),
    TONGUE: dict(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        threshold = 0.2
    ),
    UPPER_LIP: dict(
        method=GRAPH_BASED,
        alpha=1,
        beta=10,
        gamma=0,
        threshold = 0.3
    )
}
