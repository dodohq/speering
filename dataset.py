import torch
from torch.utils.data import Dataset
from os import listdir
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.data import imread
import numpy as np


class DrivingSet(Dataset):
    def __init__(self, root, training=True, transform=None):
        self.root_dir = root
        self.transform = transform
        self.files_list = []
        self.angles = []
        with open(join(self.root_dir, 'data.txt')) as dfile:
            for line in dfile:
                if line.split(',')[0] == 'center':
                    continue
                self.files_list.append(join(self.root_dir, line.split(' ')[0]))
                self.angles.append(line.split(' ')[1].strip())
        train_files, valid_files, train_angles, valid_angles = train_test_split(
            self.files_list, self.angles, test_size=0.1, random_state=100)
        self.files_list = train_files if training else valid_files
        self.angles = train_angles if training else valid_angles

    def crop_sky(self, img):
        cropped = img[60::, ::]
        return cropped

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        img = imread(self.files_list[idx])
        img = self.crop_sky(img)
        if self.transform != None:
            img = self.transform(img)
        angle = self.angles[idx]

        data = torch.from_numpy(img).float()
        label = torch.from_numpy(np.array([angle])).float()

        return (data, label)
