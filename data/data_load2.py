import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_dataloader2(path, batch_size=64, num_workers=0):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=256),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader2(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader2(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_valid=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


import random


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, ps=None):
        self.image_dir = image_dir
        self.ref_list = os.listdir(os.path.join(self.image_dir, 'ref_imgs'))
        self.image_list = []
        self.get_img_path()

        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.ps = ps

    def get_img_path(self):
        namehazedir = ["outputs10", "outputs15", "outputs20", "outputs25", "outputs30"]
        for ref_name in self.ref_list:
            haze_name = ref_name.split(".")[0] + "_synt.jpg"
            fullpath_ref = os.path.join(self.image_dir, "ref_imgs", ref_name)
            for dir in namehazedir:
                fullpath_haze = os.path.join(self.image_dir, dir, haze_name)
                self.image_list.append((fullpath_ref, fullpath_haze))


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx][1]).convert('RGB')
        label = Image.open(self.image_list[idx][0]).convert('RGB')
        ps = self.ps

        if self.ps is not None:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh - ps)
            cc = random.randint(0, ww - ps)

            image = image[:, rr:rr + ps, cc:cc + ps]
            label = label[:, rr:rr + ps, cc:cc + ps]

            if random.random() < 0.5:
                image = image.flip(2)
                label = label.flip(2)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        if self.is_test:
            name = os.path.basename(self.image_list[idx][1]) + os.path.split(self.image_list[idx][1])[-2]
            # name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits0 = x[0].split('.')
            splits1 = x[1].split('.')
            if splits0[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

            if splits1[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
