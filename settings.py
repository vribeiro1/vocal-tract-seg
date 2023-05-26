import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTSPEECH_DIR = os.path.join(BASE_DIR, "data_ArtSpeech", "training")

TRAIN = "train"
VALID = "validation"
TEST = "test"


class ArtSpeechConfig:
    RES = 136
    PIXEL_SPACING = 1.6176470518112
    FRAMERATE = 50
    SYNC_SHIFT = 2


class GottingenConfig:
    RES = 136
    PIXEL_SPACING = 1.4117647409439
    FRAMERATE = 55
    SYNC_SHIFT = 0


DATASET_CONFIG = {
    "artspeech": ArtSpeechConfig,
    "artspeech2": ArtSpeechConfig,
    "gottingen": GottingenConfig
}
