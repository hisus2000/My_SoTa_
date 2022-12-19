import cv2
import glob
import torch
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, root_dir, transform= None, is_test = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data, self.label, self.path = self.read_data(is_test)

    def read_data(self, is_test: bool):
        data = [] 
        label = []
        path = []
        mean_ = []
        std_ = []
        class_ids = {}
        folders = sorted(glob.glob(self.root_dir + f'/*'))
        for i, each in enumerate(folders):
            label_name = str(each.split('\\')[-1])
            class_ids[i] = label_name
            list_images = sorted(glob.glob(each+'/*'))
            # if len(list_images) <= 500:
            #     pass
            # else:
            #     list_images = list_images[:500]
            print(f'{i}:Read {label_name} folder:')
    
            for img in tqdm(list_images, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'):
                image = cv2.imread(img)
                # mean_.append(np.mean(image)/255)
                # std_.append(np.std(image)/255)
                # image = cv2.resize(image, (64,64))
                
                data.append(image)
                label.append(i)
                # if label_name == 'Fa':
                #     label.append(0)
                # else: 
                #     label.append(1)
                path.append(img)
                
        if not is_test:
            with open("./class_ids.json", "w") as f:
                json.dump(class_ids, f, indent=1)
        # print('MEAN: ',np.mean(mean_), 'STD: ',np.std(std_))
        return data, label, path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.label[idx]
        path = self.path[idx]

        image = np.array(image)
        label = np.array(label)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
        
        label = torch.from_numpy(label)
        label = label.long()
        return image, label, path
