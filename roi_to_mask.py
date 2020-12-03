import pdb

import os

from glob import glob
from torchvision import transforms

from dataset import VocalTractDataset
from read_roi import read_roi_file

DATA_DIR = "/home/vsouzari/Documents/loria/datasets/ArtSpeech_Vocal_Tract_Segmentation"

roi_fpaths = glob(os.path.join(DATA_DIR, "*", "S*", "contours", "*.roi"))

data = {}
for roi_fpath in roi_fpaths:
    roi_fname = os.path.basename(roi_fpath)
    image_id = roi_fname.split("_")[0]
    roi_dirname = os.path.dirname(os.path.dirname(roi_fpath))
    sequence_id = os.path.basename(roi_dirname)
    sequence_dirname = os.path.dirname(roi_dirname)
    subject_id = os.path.basename(sequence_dirname)

    if subject_id not in data:
        data[subject_id] = {}
    if sequence_id not in data[subject_id]:
        data[subject_id][sequence_id] = {}
    if image_id not in data[subject_id][sequence_id]:
        data[subject_id][sequence_id][image_id] = []

    data[subject_id][sequence_id][image_id].append(roi_fpath)

topil = transforms.ToPILImage()
for subject_id, sequences in data.items():
    for sequence_id, images in sequences.items():
        for image_id, image_fpaths in images.items():
            dirname = os.path.dirname(os.path.dirname(image_fpaths[0]))
            masks_dirname = os.path.join(dirname, "masks")
            if not os.path.exists(masks_dirname):
                os.makedirs(masks_dirname)

            for image_fpath in image_fpaths:
                roi_name, roi = list(read_roi_file(image_fpath).items())[0]

                img_fpath = os.path.join(masks_dirname, roi_name + ".png")
                if os.path.exists(img_fpath):
                    # Target mask was already generated
                    continue

                mask_tensor = VocalTractDataset.roi_to_mask_tensor(roi, (136, 136))
                mask_img = topil(mask_tensor)
                mask_img.save(img_fpath)
