import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms
import torch

# file_list = 'files_train_sampled.csv'
# label = 'labcmp'
# parent_folder = 'data'
# image_path = os.path.join(parent_folder,'image_dir')
# mask_path = os.path.join(parent_folder,(label+'_mask_dir'))

class TokaidoDataset(Dataset):
    def __init__(self, image_dir, mask_dir,is_transform=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        if is_transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((360,640)),
                transforms.Normalize(
                    mean = [0.0,0.0,0.0],
                    std=[1.0,1.0,1.0],
                )
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.masks = self.images = os.listdir(self.mask_dir)
        self.images = [i.replace('.bmp','_Scene.png') for i in self.images]


    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_path = os.path.join(self.image_dir,self.images[index])
        mask_path = os.path.join(self.mask_dir,self.masks[index])
        image = np.array(Image.open(image_path).convert('RGB'), dtype = np.float32)/255
        image = self.transform(image)
        mask = torch.Tensor(np.array(Image.open(mask_path).convert('L'), dtype = np.long))

        return image, mask

    def getImage(self,index):
        image_path = os.path.join(self.image_dir,self.images[index])
        return np.array(Image.open(image_path).convert('RGB'))
