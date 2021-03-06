#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:35:47 2021

@author: 1517suj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from legoData_1Cam import legoDataOneCamera
from torchvision import transforms
from torch.utils.data import DataLoader

##########################
### SETTINGS
##########################

# Path to weights file
PATH = "attempt_4.pth"
DATA_LOCATION = 'split_data_2'


# define basic blocks (or the "bottleneck")

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(block, self).__init__()
        
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x = x + identity # Skip connection
        x = self.relu(x)
        
        return x
            


# In[11]:


class ResNet(nn.Module): # layers (a list) is used to make different type of ResNet
    def __init__(self, block, layers, image_channels, num_classes): # image_channels: 1 for grayscale, 3 for colored image
        super(ResNet, self).__init__()
        
        # this is just the first layer, this layer does not have any residual block
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        # ResNet layers (4 blocks as shown in the paper)
        self.layer1 = self.make_layer(block, layers[0], out_channels = 64, stride = 1)
        self.layer2 = self.make_layer(block, layers[1], out_channels = 128, stride = 2)
        self.layer3 = self.make_layer(block, layers[2], out_channels = 256, stride = 2)
        self.layer4 = self.make_layer(block, layers[3], out_channels = 512, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)

        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
        

        
    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1, 
                                                          stride = stride),
                                                nn.BatchNorm2d(out_channels*4))
            
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4
        
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels)) # 256 -> 64 -> 64*4 (256) again
            
        return nn.Sequential(*layers)
        


# In[12]:
image_channels = 3
num_classes = 50


def ResNet50(img_channels, num_classes):
    return ResNet(block, [3,4,6,3], image_channels, num_classes)


model = ResNet50(image_channels, num_classes)

# model = model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# Hyperparameters
RANDOM_SEED = 17
BATCH_SIZE = 32
IMAGE_SIZE = (256,256)

# Architecture
# image_channels = 3 # image channel, grayscale -> 1, colored -> 3
DEVICE = 'cuda' 

model.to(DEVICE)

resize_transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Resize(IMAGE_SIZE),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                       ])

test_dataset = legoDataOneCamera(mode='test', dataset_root=DATA_LOCATION, transform=resize_transform, target_transform=None)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE,
                         num_workers=4,
                         shuffle=False)



def compute_accuracy(model, data_loader):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


with torch.set_grad_enabled(False): # save memory during inference
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))
