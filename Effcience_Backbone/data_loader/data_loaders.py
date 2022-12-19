import sys
import pathlib
import cv2

ROOT_DIR = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT_DIR))

from torchvision import datasets, transforms
from data_loader.customdataset import CustomDataset
from base.base_data_loader import BaseDataLoader

import numpy as np
from PIL import Image as pil_Image

class CustomDataLoader(BaseDataLoader):
    """
    CUSTOM data loading using BaseDataLoader
    """

    def __init__(
        self,
        data_dir,
        train_image_size_,
        batch_size,
        mean,
        std,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
        num_channel: int = 3,
        is_test= False,
        my_transforms=...,
    ):

        self.trsfm = transforms.Compose(
            [
                # NumpyImagePaddingSquare(num_channel, train_image_size_),
                transforms.Grayscale(num_channel),
                transforms.Resize((train_image_size_, train_image_size_)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)

            ]
        )

        self.data_dir = data_dir
        self.dataset = CustomDataset(root_dir=data_dir, transform= self.trsfm, is_test=is_test)

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
import torch
from tqdm import tqdm        
def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _, path in tqdm(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

def main():
    dataloader = CustomDataLoader(
        data_dir= '/mnt/sdb2/home/Tim_TrieuTran/Jipal/classification/F13_Defect_Classification/datasets/train',
        train_image_size_= 64,
        batch_size= 1,
        mean=0,
        std =0,
    )
    mean, std = get_mean_and_std(dataloader=dataloader)
    print(mean, std)

if __name__ == "__main__":
    main()