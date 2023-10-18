import argparse
import funcy
import numpy as np
import os
import torch
import yaml

from glob import glob
from helpers import sequences_from_dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from vt_tools.bs_regularization import regularize_Bsplines
from vt_tracker.border_segmentation import detect_borders
from vt_tracker.input import InputLoaderMixin, imagenet_normalize
from vt_tracker.postprocessing.calculate_contours import calculate_contour

from helpers import set_seeds


class MRDataset(Dataset, InputLoaderMixin):
    def __init__(self, datadir, image_dir, image_ext, sequences_dict=None):
        super().__init__()

        sequences_dict = sequences_dict or {}
        self.sequences = set(sequences_from_dict(datadir, sequences_dict))

        filepaths = funcy.lfilter(
            self._sequence_filter,
            glob(os.path.join(
                datadir,
                "*",  # subject identifier
                "*",  # sequence identifier
                image_dir,
                f"*.{image_ext}"
            ))
        )
        self.data = sorted(filepaths)
        self.image_dir = image_dir
        self.image_ext = image_ext
        self.totensor = transforms.ToTensor()

    def _sequence_filter(self, filepath):
        sequence_dir = os.path.dirname(os.path.dirname(filepath))
        sequence = os.path.basename(sequence_dir)
        subject = os.path.basename(os.path.dirname(sequence_dir))

        return (subject, sequence) in self.sequences


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        filepath = self.data[index]

        dirname = os.path.dirname(os.path.dirname(filepath))
        basename = os.path.basename(filepath)
        frame, _ = basename.split(".")
        frame_number = int(frame)
        sequence = os.path.basename(dirname)
        subject = os.path.basename(os.path.dirname(dirname))

        frame_m1 = "%04d" % (frame_number - 1)
        filepath_m1 = os.path.join(os.path.dirname(filepath), frame_m1 + f".{self.image_ext}")
        if not os.path.exists(filepath_m1):
            filepath_m1 = None

        frame_p1 = "%04d" % (frame_number + 1)
        filepath_p1 = os.path.join(os.path.dirname(filepath), frame_p1 + f".{self.image_ext}")
        if not os.path.exists(filepath_p1):
            filepath_p1 = None

        image = self.load_input(filepath_m1, filepath, filepath_p1, mode="rgb")
        image = self.totensor(image)
        image = imagenet_normalize(image)

        info = {
            "subject": subject,
            "sequence": sequence,
            "frame": frame
        }

        return info, image


def main(
    datadir,
    image_dir,
    image_ext,
    batch_size,
    num_workers=0,
    sequences_dict=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MRDataset(
        datadir,
        image_dir,
        image_ext,
        sequences_dict
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=set_seeds
    )

    progress_bar = tqdm(dataloader, desc="Processing")
    for info, batch in progress_bar:
        batch = batch.to(device)
        detected = detect_borders(batch, device.type)

        for i, item in enumerate(detected):
            subject = info["subject"][i]
            sequence = info["sequence"][i]
            frame = info["frame"][i]

            save_contours_dir = os.path.join(
                datadir,
                subject,
                sequence,
                "inference_contours"
            )
            if not os.path.exists(save_contours_dir):
                os.makedirs(save_contours_dir)

            for articulator_name, detection in item.items():
                if detection is None:
                    continue

                save_filepath = os.path.join(save_contours_dir, f"{frame}_{articulator_name}.npy")
                if os.path.exists(save_filepath):
                    continue

                mask = detection["mask"]
                contour = calculate_contour(articulator_name, mask)
                if len(contour) == 0:
                    continue
                reg_x, reg_y = regularize_Bsplines(contour, degree=2)
                reg_contour = np.array([reg_x, reg_y]).T

                np.save(save_filepath, reg_contour)


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", dest="cfg_filepath")
        args = parser.parse_args()

    with open(args.cfg_filepath) as f:
        cfg = yaml.safe_load(f.read())

    main(**cfg)
