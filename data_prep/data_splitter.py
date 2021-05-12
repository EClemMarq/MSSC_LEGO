# This file was created to split the LEGO dataset into training, validation,
# and test datasets.

# import the important libraries
import os,sys
from shutil import copy
from random import sample, seed

#%% SETTINGS -----------------------------------------------------------------
# Define the percentage of original dataset used for validation, testing.
# Everything left is used for training.
VAL_PERCENT = 10
TEST_PERCENT = 15

# Set random seed for repeatability
R_SEED = 2
seed(R_SEED)

# Location of the original dataset and the desired location for divided dataset
inDataPath = "raw_dataset"
outDataPath = "split_data_" + str(R_SEED)

# Location of file listing all the data classes
textClassList = "classNames.txt"

#%% FUNCTION DECLARATIONS ----------------------------------------------------
def dataList(fileName=None):
    #check that a file name is provided
    if fileName is None:
        sys.exit("File name is absent: [dataList(fileName=None)]")
    
    # check that file name is valid
    if validPath(fileName) is False:
        sys.exit("Provided file does not exist: [dataList(fileName=None)]")
        
    #open the file for reading only
    txtFile = open(fileName,"r")

    # use list comprehension to get the list of all videos
    dataList = [dataList.strip("\n").split(" ")[0] for dataList in txtFile]

    # Debugging: helps you see how many records are available for reading
    print("File name:{}   Lines of data:{:,} ".format(fileName,len(dataList)))
    return dataList

def validPath(dirName=None):
    
    retval = False
    if dirName is None:
        sys.exit("Path is not availabe: [dirAvailabe(dirName=None)]".format())
    
    else:        
        if not os.path.exists(dirName): retval = False
        else: retval = True
    
    return retval

def num2Str(imgNum=None):
    # Generate a three-digit number string for this image
    if imgNum < 10:
        numStr = "00" + str(imgNum)
    elif imgNum < 100:
        numStr = "0" + str(imgNum)
    else:
        numStr = str(imgNum)
        
    return numStr

def copy2SplitDir(splitDir=None,classList=None,splitNums=None):
    # Iterate over each class
    for className in classList:
        classDir = os.path.join(splitDir,className)
        
        # Make the folder, if needed
        if not os.path.exists(classDir):
            os.makedirs(classDir)  
        
        # Iterate over each number in the validation set
        for imgNum in splitNums:
            
            numStr = num2Str(imgNum)
            
            # Generate filenames for the left and right camera angles
            imgR = className + "_" + numStr + "R.png"
            imgL = className + "_" + numStr + "L.png"
            
            imgROut = os.path.join(classDir,imgR) 
            imgLOut =  os.path.join(classDir,imgL)
            
            # Copy the files if they don't already exist
            if not os.path.exists(imgROut):
                copy(os.path.join(inDataPath,imgR), imgROut)
            if not os.path.exists(imgLOut):
                copy(os.path.join(inDataPath,imgL), imgLOut)
            
    return

# If the outDataPath doesn't exist, make it
if not os.path.exists(outDataPath):
    os.makedirs(outDataPath)

# Perform list comprehension to get the class names
classList = dataList(textClassList)

#%% RANDOM SELECTION ----------------------------------------------------------
# The original dataset contains 800 images of each type of piece (000L-399R).
# To allow for future options, the matching L & R images will be stored in the
# same split (train, validation, or test).

# Generate random numbers for the files that will be moved to test/validation 
# folders

selNums = sample(range(400),400*(VAL_PERCENT+TEST_PERCENT)//100)
valNums = sorted(selNums[:(VAL_PERCENT*400//100)])
testNums = sorted(selNums[(VAL_PERCENT*400//100):])

# Everything else will be used for training
trainNums = list(range(400))
trainNums = [ele for ele in trainNums if ele not in selNums]

# Save the numbers of files used for each split
label_file = os.path.join(outDataPath,'split_nums_' + str(R_SEED)+ '.txt')

# If the file doesn't already exist, save it for future reference
if not os.path.exists(label_file):
    with open(label_file, 'w') as f:
        f.writelines("Validation Images: ---------------------- \n")
        for num in valNums:
            f.writelines(str(num) + '\n')
        f.writelines("Test Images: ---------------------------- \n")
        for num in testNums:
            f.writelines(str(num) + '\n')

#%%  COPY VALIDATION IMAGES ---------------------------------------------------

# Make the folder
valDir = os.path.join(outDataPath,"validation")
if not os.path.exists(valDir):
    os.makedirs(valDir)   
    
copy2SplitDir(splitDir=valDir,classList=classList,splitNums=valNums)

#%% COPY TEST IMAGES ----------------------------------------------------------

# Make the folder
testDir = os.path.join(outDataPath,"test")
if not os.path.exists(testDir):
    os.makedirs(testDir)

copy2SplitDir(splitDir=testDir,classList=classList,splitNums=testNums)

#%% COPY TRAIN IMAGES ---------------------------------------------------------
         
# Make the folder
trainDir = os.path.join(outDataPath,"train")
if not os.path.exists(trainDir):
    os.makedirs(trainDir)

copy2SplitDir(splitDir=trainDir,classList=classList,splitNums=trainNums)      
        
