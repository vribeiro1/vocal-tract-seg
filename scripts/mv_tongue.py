import pdb

import re
import os
import zipfile
import shutil

from glob import glob

BASE_DIR = "/home/vsouzari/Documents/Loria/datasets/data_ArtSpeech/training"

zips = glob(os.path.join(BASE_DIR, "*", "S*.zip"))

for zip_fpath in zips:
    extract_to = os.path.join(zip_fpath.rsplit(".", maxsplit=1)[0], "contours")
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_fpath, "r") as zf:
        zf.extractall(extract_to)

    rename_files = filter(lambda s: re.search(r"[0-9]+-[0-9]+-[0-9]", s), os.listdir(extract_to))
    for fname in rename_files:
        keep = os.path.basename(fname).split("-")[0]
        new_name = keep + "_tongue.roi"

        fpath = os.path.join(extract_to, fname)
        new_fpath = os.path.join(extract_to, new_name)
        shutil.move(fpath, new_fpath)
