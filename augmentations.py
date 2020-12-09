import funcy
import torch
import torchvision.transforms.functional as F

from torchvision import transforms


class MultiCompose(transforms.Compose):
    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs


class MultiRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, *imgs):
        if torch.rand(1) < self.p:
            return funcy.lmap(F.hflip, imgs)
        return imgs


class MultiRandomVerticalFlip(transforms.RandomVerticalFlip):
    def forward(self, *imgs):
        if torch.rand(1) < self.p:
            return funcy.lmap(F.vflip, imgs)
        return imgs


class MultiRandomRotation(transforms.RandomRotation):
    def forward(self, *imgs):
        angle = self.get_params(self.degrees)
        return funcy.lmap(
            lambda img: F.rotate(img, angle, self.resample, self.expand, self.center, self.fill),
            imgs
        )
