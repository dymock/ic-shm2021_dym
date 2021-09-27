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

# def reduce_res(img, scale):
#     new_width = int(img.shape[1]*scale)
#     new_height = int(img.shape[0]*scale)
#     tr = A.Resize(width = new_width,height = int(1080/4))


def populate_folder():
    df = pd.read_csv(file_list)

    for i, row in tqdm(df.iterrows(), total = df.shape[0]):
        if (os.path.isfile(row[label])):
            filename=row[label][(row[label].find(label)+len(label)+1):]
            filepath=os.path.join(mask_path,filename)
            shutil.copy(row[label],filepath)
            filename=row['image'][(row['image'].find('train')+len('train')+1):]
            filepath=os.path.join(image_path,filename)
            shutil.copy(row['image'],filepath)

def reset_folders():
    if os.path.isdir(parent_folder): shutil.rmtree(parent_folder)
    os.makedirs(image_path)
    os.makedirs(mask_path)


# tr = A.Resize(width = int(1920/4),height = int(1080/4))
# im = Image.open(r'.\img_syn_raw\train\image_case85_frame14_Scene.png')
# tr(image=im)
# lab = np.array(Image.open(r'.\synthetic\train\labcmp\image_case160_frame31.bmp').convert('RGB'))
# image.shape[1]
#
# image = np.array(Image.open(r'.\img_syn_raw\train\image_case85_frame14_Scene.png').convert('RGB'))
# image
# plt.imshow(image)
# plt.imshow(lab*40)
# tr_im = tr(image=image)
#
# np.array(tr_im)
# type(image)
# type(tr_im)
# plt.imshow(tr_im['image'])
