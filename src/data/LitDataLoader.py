from typing import Union

import lightning as L
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.augmentations.transforms import Normalize
from albumentations.augmentations.geometric.resize import Resize


from .Dataset import BuildDatasetImage


class LitDataLoader(L.LightningDataModule):
    def __init__(
        self,
        train_df: str = None,
        valid_df: str = None,
        test_df: str = None,
        path_to_image: str = None,
        path_to_test_image: str = None,
        batch_size: int = 20,
        num_workers: int = 4,
        resize: Union[None, Resize] = None,
        normalize: Union[None, Normalize] = None,
        transformer: Union[None, A.Compose] = None,
        tr_chance: float = 0.6,
    ):
        """
        Parameters:
            train_path (str): path for with csv with train dataset
            valid_path (str): path for with csv with valid dataset
            test_path (str): path for with csv with test dataset
            batch_size (int): bach size for DataLoader
            num_workers (int) num_workers for DataLoader
        """
        super().__init__()

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.path_to_image = path_to_image
        self.path_to_test_image = path_to_test_image
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize = resize
        self.normalize = normalize
        self.transformer = transformer
        self.tr_chance = tr_chance

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage is None or stage in "fit":
            self.train_data = BuildDatasetImage(
                self.train_df,
                path_to_image=self.path_to_image,
                normalize=self.normalize,
                resize=self.resize,
                transformer=self.transformer,
                tr_chance=self.tr_chance,
            )

            self.valid_data = BuildDatasetImage(
                self.valid_df,
                path_to_image=self.path_to_image,
                normalize=self.normalize,
                resize=self.resize,
                transformer=None,
                tr_chance=self.tr_chance,
            )
        if stage == "test":
            self.test_data = BuildDatasetImage(
                self.test_df,
                path_to_image=self.path_to_test_image,
                normalize=self.normalize,
                resize=self.resize,
                transformer=None,
                tr_chance=self.tr_chance,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )
