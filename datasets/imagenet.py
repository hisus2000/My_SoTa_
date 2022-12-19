from torchvision.datasets import ImageFolder
from typing import Optional, Callable
from pathlib import Path
import os

class ImageNet(ImageFolder):
    _, dirs, _ = next(os.walk("./images"))   
    WNIDS = dirs
    CLASSES = ["defect","good"]

    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        assert split in ('train', 'val')
        split_folder = Path(root) / split

        super().__init__(split_folder, transform=transform, target_transform=target_transform)
        
        self.classes = self.WNIDS
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    imagenet = ImageNet('./data_set/train', split='val')
    dataloader = DataLoader(imagenet, batch_size=4)
    print(len(imagenet))
    print(len(dataloader))