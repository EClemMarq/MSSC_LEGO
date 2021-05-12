#!/usr/bin/env python
# coding: utf-8

# # MSCS 6250 CLASS PROJECT -- IMAGE CLASSIFICATION USING ResNet50

# In[5]:


import torch
import torch.nn as nn

import time
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from os.path import exists
from legoData_1Cam import legoDataOneCamera


import matplotlib.pyplot as plt

# In[6]:


print("Available GPUs: " + str(torch.cuda.device_count())) # how many GPUs can be used


# In[7]:


# torch.cuda.get_device_name(0) # GPU's type

# In[9]:


##########################
### SETTINGS
##########################

# File organization
ATTEMPT_NUMBER = 4
DATA_LOCATION = 'split_data_2'

# Generate a preliminary path to the results file
RESULTS_FILE = "Results/attempt_" + str(ATTEMPT_NUMBER)

# Verify that this results file doesn't already exist. If it does, update the
# attempt number
while exists(RESULTS_FILE):
    print("A folder for attempt_" + str(ATTEMPT_NUMBER) + " already exists.")
    ATTEMPT_NUMBER += 1
    print("Creating a new folder named attempt_" + str(ATTEMPT_NUMBER) + " for this attempt." )
    RESULTS_FILE = "Results/attempt_" + str(ATTEMPT_NUMBER)

# Hyperparameters
RANDOM_SEED = 17
LEARNING_RATE = 0.00001
NUM_EPOCHS = 40
BATCH_SIZE = 32
IMAGE_SIZE = (256,256)

# Architecture
NUM_CLASSES = 50
image_channels = 3 # image channel, grayscale -> 1, colored -> 3
DEVICE = 'cuda' 

##########################
### DEFINE DATA TRANSFORMS
##########################

resize_transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Resize(IMAGE_SIZE),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                       ])

#####################################################
### Data Loaders
#####################################################


train_dataset = legoDataOneCamera(mode='train', dataset_root=DATA_LOCATION, transform=resize_transform, target_transform=None)
valid_dataset = legoDataOneCamera(mode='validation', dataset_root=DATA_LOCATION, transform=resize_transform, target_transform=None)
test_dataset = legoDataOneCamera(mode='test', dataset_root=DATA_LOCATION, transform=resize_transform, target_transform=None)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=False)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE,
                         num_workers=4,
                         shuffle=False)

#####################################################

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

for images, labels in test_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break
    
for images, labels in valid_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break


# ResNet 50

# In[10]:


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


def ResNet50(img_channels, num_classes):
    return ResNet(block, [3,4,6,3], image_channels, num_classes)


# In[14]:

# Display images for visualization

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    # Convert from tensor to np array
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.tick_params(
        axis='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False)
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# Show the images for visualization
# NOTE: This works best for BATCH_SIZE <= 16
# imshow(make_grid(images,nrow=4,padding=4))


torch.manual_seed(RANDOM_SEED)

##########################
### SUMMARY WRITER AND OPTIMIZER
##########################

model = ResNet50(image_channels, NUM_CLASSES)

writer = SummaryWriter(RESULTS_FILE)
writer.add_graph(model, images)


model.to(DEVICE)

# count the total trainable parameters in the network

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# In[22]:


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


def compute_epoch_loss(model, data_loader):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            logits, probas = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss

start_time = time.time()
for epoch in range(NUM_EPOCHS):
    
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), cost))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        train_acc = compute_accuracy(model, train_loader)
        valid_acc = compute_accuracy(model, valid_loader)
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
              epoch+1, NUM_EPOCHS, train_acc, valid_acc))
        
        writer.add_scalars('Accuracy', {'training': train_acc, 
                                        'validation': valid_acc},epoch)
        
        cost = compute_epoch_loss(model, train_loader)
        
        writer.add_scalar('Loss', cost, epoch)        

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
    if not epoch % 5:
        writer.flush()
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

writer.close()


# In[23]:


##########################
### EVALUATION ON TEST SET
##########################

# NOTE: This should only be performed once all the hyperparameters have been
# tuned using the validation set. (Performed once only.)

# with torch.set_grad_enabled(False): # save memory during inference
#     print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))




