## OVERVIEW

This repository contains the code and data for our final project for MSSC 6250.  In this project, we train a convolutional neural network (CNN) to classify 3D renderings of LEGO bricks.  The dataset of LEGO bricks was created by Joost Hazelzet (1).  Our convolutional neural network uses the ResNet50 architecture (2).

This repository contains all our pre-processing code, our network architecture and training scheme, and results from several of our attempts to train the network.  This README will help you understand our methods and how to replicate them.

## DATA PREPROCESSING

### DOWNLOAD DATA

Data was obtained from the "dataset" folder in (1). This folder contains all 40k images in one folder, with no subfolders.  We called this the "raw dataset".  To simplify the partitioning process, all the spaces in the filenames were then replaced with underscores using the following bash command inside the downloaded dataset folder:

> $ for file in *; do mv "$file" `echo $file | tr ' ' '_'` ; done

### PARTITION DATA

In the _data\_prep_ folder, there is a script called data\_splitter.py. This script will take the data from its originally downloaded organization and create a copy where images are randomly assigned to a training,testing, or validation split. This script requires the classNames.txt file that also appears in the _data\_prep_ folder.  This also requires that all spaces in filenames be replaced with underscores, as seen in the above section.

### CREATE DATALOADER

## NETWORK ARCHITECTURE

### DEFINING THE ARCHITECTURE

### SAVING MODEL WEIGHTS


## OUTPUT RESULTS


 
## REFERENCES

(1) Joost Hazelzet, "Images of LEGO Bricks." (December 31, 2019). Distributed by kaggle.com. url: https://www.kaggle.com/joosthazelzet/lego-brick-images/version/4 (accessed May 3rd, 2021).

(2) K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.
