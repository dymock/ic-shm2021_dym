import torch
import torchvision
from torch.utils.data import DataLoader
from dataset_prep import TokaidoDataset
import os

def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print('==> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer):
    print('==> loading checkpoint')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss



# def save_predictions_as_imgs(
#     loader, model, folder = 'saved_images', device = 'cpu'):
#     model.eval()
#     for idx, (x,y) in enumerate(loader):
#         x = x.to(device = device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, os.path.join(folder,f'pred_{idx}.png')
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), os.path.join(folder,f'{idx}.png'))
#
#
#     model.train()
