import torch
import torchvision
from torch.utils.data import DataLoader
from dataset_prep import TokaidoDataset
import os
import numpy as np
import cv2
from label_color_map import label_color_map as label_map
from PIL import Image

def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print('==> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer,device):
    print('==> loading checkpoint')
    checkpoint = torch.load(path,map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_epochs = checkpoint['total_epochs']
    loss = checkpoint['loss']
    steps = checkpoint['steps']

    return model, optimizer, total_epochs, loss, steps


def save_prediction(model,loader,sample_idxs,folder=r'../Tokaido_dataset/predictions',fullres=False):
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

def draw_segmentation_map(labels):
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

def image_overlay(image, segmented_image):
    # overlay the images with predicted labels
    alpha = 0.4  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image
