import torch
import torchvision.models.segmentation as segmentation
from tqdm import tqdm
import os
import sys
from utils import (load_checkpoint, save_checkpoint, save_predictions, get_ValSet, generate_predictions)
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

SAMPLE_PREDICTIONS=20
SPLIT_RATIO=0.9
FULLRES = True
DEVICE='cpu'
SEED = 0


def main():

    for MASK in MASK_CONFIGS:
        # get data for mask type
        dataset = TokaidoDataset(image_dir = TRAIN_IMG_DIR, mask_dir = MASK['train_dir'])
        val_dataset = get_ValSet(dataset,split_ratio=SPLIT_RATIO,seed=SEED)
        # val_loader = DataLoader(val_dataset)
        sample_idxs = np.arange(0, val_dataset.__len__(), int(val_dataset.__len__()/(SAMPLE_PREDICTIONS-1)))

        for MODEL in MODEL_CONFIGS:
            # load model from torchvision model lib
            model=MODEL['model'](num_classes=MASK['num_classes'])

            # get most up to date checkpoint for task model
            model_save_name = (MODEL['model'].__name__+'-' + MASK['type'] + '-checkpoint.pth.tar')
            checkpoint_path = os.path.join(MODEL_SAVE_DIR,model_save_name)
            try:
                model,_,_,_,_ = load_checkpoint(path=checkpoint_path,model=model,device=DEVICE)
                print(MASK['type']+'-'+MODEL['model'].__name__+' loaded')
            except:
                print('Error loading '+MASK['type']+'-'+MODEL['model'].__name__)

            # run model for dataset sample indices
            images, image_names = generate_predictions(model,val_dataset,sample_idxs)
            image_names = [i.replace('_Scene.png','_Prediction') for i in image_names]
            for i in range(0,len(images)):
                path=os.path.join(PRED_PARENT_DIR,MASK['type'],image_names[i])
                if not os.path.isdir(path):
                    os.makedirs(path)
                save_name = MODEL['model'].__name__ + '.png'
                images[i].save(os.path.join(path,save_name))


if __name__ =='__main__':
    main()
