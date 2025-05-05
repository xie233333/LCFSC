from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data

def get_loader(args, config, epoch):
    apply_transform = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()])
        
    transform = transforms.Compose([
        transforms.Resize([128, 128]),            
        transforms.ToTensor()])        
        
    train_dataset = torchvision.datasets.ImageFolder(root='TWC/datasets/UDIS-D-TWC/train', transform=apply_transform)
    test_dataset = torchvision.datasets.ImageFolder(root='TWC/datasets/UDIS-D-TWC/test', transform=transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        
    train_sampler.set_epoch(epoch)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, config.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
		    batch_sampler=train_batch_sampler, pin_memory=False, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_dataset,
		    batch_size=config.batch_size, sampler=test_sampler, drop_last=True, pin_memory=False, num_workers=16)

    return train_loader, test_loader

