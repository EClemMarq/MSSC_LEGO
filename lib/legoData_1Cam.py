import os
import numpy as np
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

# This class allows for the LEGO dataset to be accessed and used for a CNN model.
# It is assumed that the dataset has already been organized in the following file hierarchy:
#
# dataset_root
#|       | classNames.txt
#|       | test
#|       |   |   14719_flat_tile_corner_2x2
#|       |   |          | 14719_flat_tile_corner_2x2_xxxL.png
#|       |   |          | 14719_flat_tile_corner_2x2_xxxR.png
#|       |   |          | ...
#|       |   |   15672_roof_tile_1x2
#|       |   |          | 15672_roof_tile_1x2_xxxL.png
#|       |   |          | 15672_roof_tile_1x2_xxxR.png
#|       |   |          | ...
#|       |   |   ...
#|       | train
#|       |   |   14719_flat_tile_corner_2x2
#|       |   |          | 14719_flat_tile_corner_2x2_xxxL.png
#|       |   |          | 14719_flat_tile_corner_2x2_xxxR.png
#|       |   |          | ...
#|       |   |   15672_roof_tile_1x2
#|       |   |          | 15672_roof_tile_1x2_xxxL.png
#|       |   |          | 15672_roof_tile_1x2_xxxR.png
#|       |   |          | ...
#|       |   |   ...
#|       | validation
#|       |   |   14719_flat_tile_corner_2x2
#|       |   |          | 14719_flat_tile_corner_2x2_xxxL.png
#|       |   |          | 14719_flat_tile_corner_2x2_xxxR.png
#|       |   |          | ...
#|       |   |   15672_roof_tile_1x2
#|       |   |          | 15672_roof_tile_1x2_xxxL.png
#|       |   |          | 15672_roof_tile_1x2_xxxR.png
#|       |   |          | ...
#|       |   |   ...

class legoDataOneCamera(Dataset):
    def __init__(self, mode='train', dataset_root=None, transform=None, target_transform=None):
        self.dataset_root = dataset_root
        self.split_dir = os.path.join(dataset_root,mode)
        self.transform = transform
        self.target_transform = target_transform


        self.fnames, labels = [], []

        # This will list out all the LEGO classes in the folder
		# and iterate over each of them.
        for label in sorted(os.listdir(self.split_dir)):
		
            # This will list out all the files in the class,
            # iterate over each of them, and append the
            # label to the long list of labels
            for fname in os.listdir(os.path.join(self.split_dir,label)):    
                self.fnames.append(os.path.join(self.split_dir, label, fname))
                labels.append(label)
			
        # prepare a mapping between the label names (strings) & indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.img_labels = np.array([self.label2index[label] for label in labels], dtype=int)

		# Save a .txt file listing the numeric values for each label
        label_file = os.path.join(dataset_root,'numeric_class_labels.txt')
        if not os.path.exists(label_file):
            with open(label_file, 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id + 1) + ' ' + label + '\n')
		
    def __len__(self):
        return len(self.img_labels)
		
    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        image = read_image(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label
