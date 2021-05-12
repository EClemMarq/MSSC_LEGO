## OVERVIEW

This repository contains the code and data for our final project for MSSC 6250.  In this project, we train a convolutional neural network (CNN) to classify 3D renderings of LEGO bricks.  The dataset of LEGO bricks was created by Joost Hazelzet (1).  Our convolutional neural network uses the ResNet50 architecture (2).

This repository contains all our pre-processing code, our network architecture and training scheme, and results from several of our attempts to train the network.  This README will help you understand our methods and how to replicate them.

## DATA PREPROCESSING

### DOWNLOAD DATA

Data was obtained from the "dataset" folder in (1). This folder contains all 40k images in one folder, with no subfolders.  We called this the "raw dataset".  To simplify the partitioning process, all the spaces in the filenames were then replaced with underscores using the following bash command inside the downloaded dataset folder:

> $ for file in *; do mv "$file" `echo $file | tr ' ' '_'` ; done

### PARTITION DATA

In the *data\_prep* folder, there is a script called *data\_splitter.py*. This script will take the data from its originally downloaded organization and create a copy where images are randomly assigned to a training,testing, or validation split. This script requires the *classNames.txt* file that also appears in the *data\_prep* folder.  This also requires that all spaces in filenames be replaced with underscores, as seen in the above section.

### CREATE DATALOADER

Once the data has been properly partitioned, the Dataset class legoDataOneCamera from the file *legoData\_1Cam.py* can be used to generate the appropriate dataloader.

## NETWORK ARCHITECTURE

### DEFINING AND TRAINING THE MODEL

The model architecture is defined in the file *ResNet50\_MSSC6250\_Project.py*.  There are several options available in the Settings section that allow for specifying a path to the partitioned data, a path for the output results, a random seed (for repeatability), and model hyperparameters.  Running this script defines and trains the model on the specified partitioned data.

During training, the model accuracy on the training and validation sets, as well as the training loss, are saved using tensorboard event files.  For more details, see the "Tensorboard Results" section of this document.

### SAVING MODEL WEIGHTS AND EVAULATION ON TEST SET

Once the model has been sufficiently trained, the weights can be saved using the command:

> torch.save(model.state_dict(), {*filename*})

Once they have been saved, they can be reloaded and evaluated on the test set using the script *load_trained_model.py*

## TENSORBOARD RESULTS

During training, the model accuracy on the training and validation sets, as well as the training loss, are saved using tensorboard event files. By default, these files are stored in the path *Results/attempt_{k}*, where k is the number of the attempt.  This number can be manually set, but the *ResNet50\_MSSC6250\_Project.py* will not overwrite an existing *attempt_k* folder.  Instead, it will generate a new attempt number and create a folder using that number.

If tensorboard is installed on your device, these files can be viewed in Tensorboard using the command:

> tensorboard --logdir Results/{*attempt_name*}
 
## REFERENCES

(1) Joost Hazelzet, "Images of LEGO Bricks." (December 31, 2019). Distributed by kaggle.com. url: https://www.kaggle.com/joosthazelzet/lego-brick-images/version/4 (accessed May 3rd, 2021).

(2) K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.
