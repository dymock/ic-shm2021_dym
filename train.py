import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import lraspp_mobilenet_v3_large as MODEL
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from utils import (load_checkpoint, save_checkpoint, save_prediction)
from dataset_prep import TokaidoDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random

LEARNING_RATE = 1E-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 60
NUM_WORKERS = 8
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
NUM_CLASSES = 9
TRANSFORM_SCALE = 1
PIN_MEMORY = True
SPLIT_RATIO = 0.9
TRAIN_IMG_DIR = r'../Tokaido_dataset/img_syn_raw/train'
TRAIN_MASK_DIR = r'../Tokaido_dataset/synthetic/train/labcmp'
MODEL_SAVE_DIR = r'../Tokaido_dataset/model_save'
MODEL_SAVE_NAME = (MODEL.__name__+'-' + TRAIN_MASK_DIR[len(TRAIN_MASK_DIR)-3:] + '-checkpoint.pth.tar')
SUMMARY_WRITE_DIR = (r'../Tokaido_dataset/summary_writer/' + MODEL.__name__+'_' + TRAIN_MASK_DIR[len(TRAIN_MASK_DIR)-3:])
PREDICTIONS_DIR = (r'../Tokaido_dataset/predictions/' + MODEL.__name__+'_' + TRAIN_MASK_DIR[len(TRAIN_MASK_DIR)-3:])
VAL_MODE = False
SEED = 0
SAMPLE_PREDICTIONS=20
FULLRES = True


def train_fn(loader, model, optimizer, loss_fn, scaler, total_epochs, writer, writer_step):

    loop = tqdm(loader)
    epoch_IoU=0
    batch_num=0
    for batch_idx, (input,targets) in enumerate(loop):
        input = input.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        # Forward pass
        predictions = model(input)
        loss = loss_fn(predictions['out'], targets.long())
        batch_intersection = batch_union =0
        for i in range(0,len(targets)):
            intersection, union = get_IoU(predictions['out'][i],targets[i])
            batch_intersection += intersection
            batch_union += union
        if not intersection == 0:
            batch_IoU = batch_intersection / batch_union
        else:
            batch_IoU = 0

        # Backpropogation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item(), IoU=batch_IoU, total_epochs=total_epochs)

        writer.add_scalar('Batch IoU', batch_IoU, global_step = writer_step)
        writer.add_scalar('Batch Loss', loss, global_step = writer_step)

        writer_step+=1
        epoch_IoU+=batch_IoU
        batch_num+=1
    epoch_IoU=epoch_IoU/batch_num
    return loss, writer_step, epoch_IoU

def get_IoU(prediction,mask):
    mask = mask.squeeze().to(device='cpu').numpy()
    prediction = torch.argmax(prediction.to(device='cpu').squeeze(), dim=0).numpy()
    intersection = (prediction == mask).sum()
    union = mask.shape[0]*mask.shape[1]
    return intersection, union

def validation(loader, model):
    model.eval()
    IoU=[]
    loop = tqdm(loader)

    with torch.no_grad():
        for batch, (input,target) in enumerate(loop):
            image = input.to(device=DEVICE)
            mask = target
            prediction = model(image)
            intersection, union = get_IoU(prediction['out'], mask)

            if not union == 0:
                IoU.append([intersection, union, intersection/union])
                loop.set_postfix(IoU=intersection / union)
            else:
                IoU.append([intersection, union, 0.0])
                loop.set_postfix(IoU=0.0)
            intersection = union = 0
    sum_intersection, sum_union, sum_IoU = [sum(x) for x in zip(*IoU)]

    meanIoU = sum_intersection / sum_union

    return meanIoU, IoU

def main():

    torch.manual_seed(SEED)
    model = MODEL(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(SUMMARY_WRITE_DIR)

    if not os.path.isdir(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)
    if not os.path.isdir(SUMMARY_WRITE_DIR):
        os.mkdir(SUMMARY_WRITE_DIR)
    if not os.path.isdir(PREDICTIONS_DIR):
        os.mkdir(PREDICTIONS_DIR)

    TOTAL_EPOCHS=0
    writer_step=0
    if os.path.isfile(os.path.join(MODEL_SAVE_DIR,(MODEL.__name__ + '-checkpoint.pth.tar'))):
        model,optimizer,TOTAL_EPOCHS,loss,writer_step = load_checkpoint(os.path.join(MODEL_SAVE_DIR,(MODEL.__name__ + '-checkpoint.pth.tar')),model,optimizer,DEVICE)

    dataset = TokaidoDataset(image_dir = TRAIN_IMG_DIR, mask_dir = TRAIN_MASK_DIR)

    train_size=int(len(dataset)*SPLIT_RATIO)
    val_size=len(dataset)-train_size
    train_dataset, val_dataset = random_split(dataset,[train_size,val_size])

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 1, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = False)

    if not VAL_MODE:
        model.train()
        for epoch in range(NUM_EPOCHS):
            loss,writer_step,IoU = train_fn(train_loader, model, optimizer, loss_fn, scaler,TOTAL_EPOCHS,writer,writer_step)
            writer.add_scalar('Epoch IoU', IoU, global_step = TOTAL_EPOCHS)
            writer.add_scalar('Epoch Loss', loss, global_step = TOTAL_EPOCHS)
            TOTAL_EPOCHS+=1
            checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'total_epochs':TOTAL_EPOCHS,
            'loss':loss,
            'steps':writer_step
            }
            save_checkpoint(checkpoint, filename = os.path.join(MODEL_SAVE_DIR,(MODEL.__name__ + '-checkpoint.pth.tar')))
    else:
        meanIoU,IoU = validation(val_loader, model)
        print('meanIoU: {:.2f}'.format(meanIoU))
        np.savetxt('IoU.csv',IoU,delimiter=',')

    if (SAMPLE_PREDICTIONS > 0):
        random.seed(SEED)
        sample_idxs = random.sample(val_loader.dataset.indices,SAMPLE_PREDICTIONS)
        save_prediction(model,val_loader,sample_idxs,fullres=FULLRES,folder=PREDICTIONS_DIR)

if __name__ =='__main__':
    main()
