import torch
import torchvision.models.segmentation as segmentation
from tqdm import tqdm
import os
import sys
import utils
from dataset_prep import TokaidoDataset
from torch.utils.data import (DataLoader, random_split)
import numpy as np

MODEL_CONFIGS=[
{'model':segmentation.lraspp_mobilenet_v3_large,
'batch_size':1,
'num_workers':0},
{'model':segmentation.deeplabv3_mobilenet_v3_large,
'batch_size':1,
'num_workers':0},
{'model':segmentation.deeplabv3_resnet50,
'batch_size':1,
'num_workers':0},
{'model':segmentation.deeplabv3_resnet101,
'batch_size':1,
'num_workers':0},
]

MASK_CONFIGS =[
{'type':'cmp',
'train_dir':'../Tokaido_dataset/synthetic/train/labcmp',
'num_classes':9},
{'type':'dmg',
'train_dir':'../Tokaido_dataset/synthetic/train/labdmg',
'num_classes':4},
]
# TRAIN_IMG_DIR ='../Tokaido_dataset/img_syn_raw/train'
TRAIN_IMG_DIR = '../Tokaido_dataset/img_syn_raw/train'
MODEL_SAVE_DIR='../Tokaido_dataset/model_save'
PRED_PARENT_DIR='../Dropbox/comparison'

SAMPLE_PREDICTIONS=2
SPLIT_RATIO=0.9
FULLRES = True
DEVICE='cpu'
SEED = 0

def get_IoU(prediction,mask):
    mask = mask.squeeze().to(device='cpu').numpy()
    num_classes = prediction.shape[1]
    prediction = torch.argmax(prediction.to(device='cpu').squeeze(), dim=0).numpy()
    intersection = []
    union =[]
    for i in range(0,num_classes):
        intersection.append(((prediction == mask) & (mask == i)).sum())
        union.append(((prediction == i) ^ (mask == i)).sum())
        union[-1]+=intersection[-1]
    intersection.append((prediction == mask).sum())
    union.append(mask.shape[0]*mask.shape[1])
    return intersection, union

def main():

    for MASK in MASK_CONFIGS:
        # get data for mask type
        dataset = TokaidoDataset(image_dir = TRAIN_IMG_DIR, mask_dir = MASK['train_dir'])
        val_dataset = utils.get_ValSet(dataset,split_ratio=SPLIT_RATIO,seed=SEED)
        # val_loader = DataLoader(val_dataset)
        sample_idxs = np.arange(0, val_dataset.__len__(), int(val_dataset.__len__()/(SAMPLE_PREDICTIONS-1)))

        images, image_names = utils.get_mask(val_dataset,sample_idxs)
        image_names = [i.replace('_Scene.png','_Prediction') for i in image_names]
        for i in range(0,len(images)):
            path=os.path.join(PRED_PARENT_DIR,MASK['type'],image_names[i])
            if not os.path.isdir(path):
                os.makedirs(path)
            save_name = 'ground_truth.png'
            images[i].save(os.path.join(path,save_name))


        for MODEL in MODEL_CONFIGS:
            # load model from torchvision model lib
            model=MODEL['model'](num_classes=MASK['num_classes'])

            # get most up to date checkpoint for task model
            model_save_name = (MODEL['model'].__name__+'-' + MASK['type'] + '-checkpoint.pth.tar')
            checkpoint_path = os.path.join(MODEL_SAVE_DIR,model_save_name)
            try:
                model,_,_,_,_ = utils.load_checkpoint(path=checkpoint_path,model=model,device=DEVICE)
                print(MASK['type']+'-'+MODEL['model'].__name__+' loaded')
            except:
                print('Error loading '+MASK['type']+'-'+MODEL['model'].__name__)

            # run model for dataset sample indices
            images, image_names, IoU = utils.generate_predictions(model,val_dataset,sample_idxs)
            image_names = [i.replace('_Scene.png','_Prediction') for i in image_names]
            for i in range(0,len(images)):
                path=os.path.join(PRED_PARENT_DIR,MASK['type'],image_names[i])
                # if not os.path.isdir(path):
                #     os.makedirs(path)
                save_name = MODEL['model'].__name__ + '.png'
                images[i].save(os.path.join(path,save_name))
                IoU[i].insert(0,image_names[i])
            save_name=MODEL['model'].__name__+'_IoU.csv'
            np.savetxt(os.path.join(PRED_PARENT_DIR,MASK['type'],save_name), IoU, delimiter=',',fmt='%s')


if __name__ =='__main__':
    main()

# ANNOTATIONS:
# 1: Structural component recognition
#    1 - Nonbridge
#    2 - Slab
#    3 - Beam
#    4 - Column
#    5 - Nonstructural components (Poles, Cables, Fences)
#    6 - Rail
#    7 - Sleeper
#    8 - Other components
#
#
# 2. Damage recognition
#    1. No Damage
#    2. Concrete Damage (cracks, spalling)
#    3. Exposed Rebar
