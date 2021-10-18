import torch
import torchvision
from torch.utils.data import (DataLoader, random_split)
import torch.optim as optim
from dataset_prep import TokaidoDataset
import os
import numpy as np
import cv2
from label_color_map import label_color_map as label_map
from PIL import Image
from tqdm import tqdm

def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print('==> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer=None,device='cpu'):
    print('==> loading checkpoint')
    checkpoint = torch.load(path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr = 1E-4)
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_epochs = checkpoint['total_epochs']
    loss = checkpoint['loss']
    steps = checkpoint['steps']

    return model, optimizer, total_epochs, loss, steps

def get_ValSet(dataset,split_ratio=0.9,seed=0):
    train_size=int(len(dataset)*split_ratio)
    val_size=len(dataset)-train_size
    torch.manual_seed(seed)
    _, val_dataset = random_split(dataset,[train_size,val_size])
    return  val_dataset


def save_predictions(model,loader,sample_idxs,folder=r'../Tokaido_dataset/predictions',fullres=False):
    model.to(device='cpu')
    model.eval()
    with torch.no_grad():
        for idx, (input,target) in enumerate(loader):
            if loader.dataset.indices[idx] in sample_idxs:
                file = loader.dataset.dataset.images[loader.dataset.indices[idx]].replace('_Scene.png','_Prediction.png')
                print('Saving ',file)
                input=input.to(device='cpu')
                output = model(input)
                output = torch.argmax(output['out'].squeeze(), dim=0).to('cpu').numpy()
                image=loader.dataset.dataset.getImage(loader.dataset.indices[idx])
                prediction = draw_segmentation_map(output)
                if fullres:
                    prediction=cv2.resize(prediction,dsize=(1920,1080),interpolation=cv2.INTER_CUBIC)
                else:
                    image=cv2.resize(image,dsize=(640,360))

                image = image_overlay(image,prediction)
                image=Image.fromarray(image)
                image.save(os.path.join(folder,file))
    model.train()

def get_mask(dataset, sample_idxs, fullres = False):
    images=[]
    image_names=[]
    for idx in sample_idxs:
        _,mask = dataset.__getitem__(idx)
        image=dataset.dataset.getImage(dataset.indices[idx])
        mask = draw_segmentation_map(mask)
        if fullres:
            mask=cv2.resize(mask,dsize=(1920,1080),interpolation=cv2.INTER_CUBIC)
        else:
            image=cv2.resize(image,dsize=(640,360))
        image = image_overlay(image,mask)
        images.append(Image.fromarray(image))
        image_names.append(dataset.dataset.images[dataset.indices[idx]])
    return images, image_names

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
    IoU = [a / b for a, b in zip(intersection, union)]

    return IoU

def generate_predictions(model,dataset,sample_idxs,fullres=False):
    model.to(device='cpu')
    model.eval()
    images=[]
    image_names=[]
    IoU=[]
    with torch.no_grad():
        for idx in sample_idxs:
            print('Generating image ', dataset.indices[idx])
            input,mask = dataset.__getitem__(idx)
            input = input.unsqueeze(0).to(device='cpu')
            output = model(input)
            IoU.append(get_IoU(output['out'],mask))
            output = torch.argmax(output['out'].squeeze(), dim=0).to('cpu').numpy()
            image=dataset.dataset.getImage(dataset.indices[idx])
            prediction = draw_segmentation_map(output)
            if fullres:
                prediction=cv2.resize(prediction,dsize=(1920,1080),interpolation=cv2.INTER_CUBIC)
            else:
                image=cv2.resize(image,dsize=(640,360))
            image = image_overlay(image,prediction)
            images.append(Image.fromarray(image))
            image_names.append(dataset.dataset.images[dataset.indices[idx]])


    model.train()
    return images, image_names, IoU

def draw_segmentation_map(labels): #thank you Zhipeng > https://github.com/anucecszl/CV_competition
    # create the RGB version of labels for model outputs
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

def image_overlay(image, segmented_image):  #thank you Zhipeng > https://github.com/anucecszl/CV_competition
    # overlay the images with predicted labels
    alpha = 0.4  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image
