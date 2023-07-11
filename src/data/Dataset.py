from typing import Union
import random

import torch
import numpy as np
import albumentations as A
import cv2
from pandas import DataFrame
from torch.utils.data import Dataset
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.transforms import (
    Normalize,
    RandomBrightnessContrast,
    RandomFog,
)
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BuildDatasetImage(Dataset):
    def __init__(
        self,
        df: DataFrame,
        path_to_image: str,
        normalize: Union[None, Normalize] = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
        resize: Union[None, Resize] = Resize(512, 720),
        transformer: Union[None, A.Compose] = A.Compose(
            [
                A.Rotate(limit=80, p=0.9, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.6),
                A.VerticalFlip(p=0.5),
                RandomFog(),
                RandomBrightnessContrast(),
            ]
        ),
        tr_chance: float = 0.6,
    ):
        super().__init__()
        self.df = df
        self.path_to_image = path_to_image
        self.normalize = normalize
        self.resize = resize
        self.transformer = transformer
        self.tr_chance = tr_chance

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = self.df.loc[index, "image"]
        label = self.df.loc[index].to_list()[1:]
        img_x = Image.open(f"{self.path_to_image}/{image}")
        img_x = np.array(img_x.convert("RGB"), dtype="float32")

        label_sofa = float(label[0])
        label_wardrobe = float(label[1])
        label_chair = float(label[2])
        label_armchair = float(label[3])
        label_table = float(label[4])
        label_commode = float(label[5])
        label_bed = float(label[6])

        if self.resize:
            img_x = self.resize(image=img_x)["image"]

        if self.normalize:
            img_x = self.normalize(image=img_x)["image"]

        if self.transformer and (random.uniform(0, 1) < self.tr_chance):
            img_x = self.transformer(image=img_x)["image"]

        img_x = torch.from_numpy(img_x).permute(2, 0, 1)

        return {
            "img_x": img_x,
            "labels": {
                "sofa": label_sofa,
                "wardrobe": label_wardrobe,
                "chair": label_chair,
                "armchair": label_armchair,
                "table": label_table,
                "commode": label_commode,
                "bed": label_bed,
            },
        }
