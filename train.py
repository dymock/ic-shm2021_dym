import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large as MODEL
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from utils import (load_checkpoint, save_checkpoint)
from dataset_prep import TokaidoDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np


LEARNING_RATE = 1E-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
NUM_EPOCHS = 4
NUM_WORKERS = 4
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
NUM_CLASSES = 9
TRANSFORM_SCALE = 1
PIN_MEMORY = True
SPLIT_RATIO = 0.9
TRAIN_IMG_DIR = r'../Tokaido_dataset/img_syn_raw/train'
TRAIN_MASK_DIR = r'../Tokaido_dataset/synthetic/train/labcmp'
MODEL_SAVE_DIR = r'../Tokaido_dataset/model_save'
SUMMARY_WRITE_DIR = r'../Tokaido_dataset/summary_writer'
PREDICTIONS_DIR = r'../Tokaido_dataset/predictions'

#
# dataset.images[1]
# dataset.image_dir
# from PIL import Image
#
# Image.open(os.path.join(dataset.image_dir,dataset.images[0]))

def train_fn(loader, model, optimizer, loss_fn, scaler):

    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions['out'], targets.long())

        # Backpropogation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

    return loss

def check_accuracy(loader, model):

    model.eval()
    IoU=[]



    loop = tqdm(val_loader)

    with torch.no_grad():
        for batch,data in enumerate(loop):
            image = data[0]
            mask = data[1].squeeze().numpy()
            predictions = model(image)
            predictions = predictions['out']
            predictions = torch.argmax(predictions.squeeze(), dim=0).numpy()

            union=intersection=0
            for i in range(0,len(predictions)):
                for j in range(0,len(predictions[i])):
                    if predictions[i,j]==mask[i,j]:
                        intersection+=1
                        union+1
                    else:
                        union+=1
            IoU.append([intersection, union])
            loop.set_postfix(IoU=intersection / union)
            intersection = union = 0
    sum_intersection, sum_union, sum_IoU = [sum(x) for x in zip(*IoU)]

    return meanIoU, IoU


def transform():
    transform = transforms.Compose([
        transforms.Resize(width = IMAGE_WIDTH, height = IMAGE_HEIGHT),
        transforms.Normalize(
            mean = [0.0,0.0,0.0],
            std=[1.0,1.0,1.0],
            max_pixel_value=255.0,),
    ])
    return transform

def main():

    model = MODEL(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)
    if not os.path.isdir(SUMMARY_WRITE_DIR):
        os.mkdir(SUMMARY_WRITE_DIR)
    if not os.path.isdir(PREDICTIONS_DIR):
        os.mkdir(PREDICTIONS_DIR)

    if os.path.isfile(os.path.join(MODEL_SAVE_DIR,(MODEL.__name__ + '-checkpoint.pth.tar'))):
        model,optimizer,_,_ = load_checkpoint(os.path.join(MODEL_SAVE_DIR,(MODEL.__name__ + '-checkpoint.pth.tar')),model,optimizer)

    dataset = TokaidoDataset(image_dir = TRAIN_IMG_DIR, mask_dir = TRAIN_MASK_DIR)

    train_size=int(len(dataset)*SPLIT_RATIO)
    val_size=len(dataset)-train_size
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True)
    val_loader = DataLoader(train_dataset, batch_size = 1, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True)


    for epoch in range(NUM_EPOCHS):
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch':epoch,
            'loss' :loss
        }
        save_checkpoint(checkpoint, filename = os.path.join(MODEL_SAVE_DIR,(MODEL.__name__ + '-checkpoint.pth.tar')))

    meanIoU,IoU = check_accuracy(val_loader, model)
    print('meanIoU: '+meanIoU)
    np.savetxt('IoU.csv',IoU,delimiter='.')

if __name__ =='__main__':
    main()
