import torch
from torch.utils.data import Dataset
from os import listdir, walk
from os.path import join
from sklearn.model_selection import train_test_split
from skimage.data import imread
from skimage.transform import resize
import numpy as np
import re


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
                self.angles.append(float(line.split(' ')[1].strip()))
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
        img = img.transpose(2, 0, 1)
        if self.transform != None:
            img = self.transform(img)
        angle = self.angles[idx]

        data = torch.from_numpy(img).float()
        label = torch.from_numpy(np.array([angle])).float()

        return (data, label)


class DodoSet(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.files = []
        for r, _, files in walk(self.root_dir):
            for f in files:
                if f.endswith('.jpg'):
                    self.files.append(join(r, f))

    def __len__(self):
        return len(self.files)

    def crop_sky(self, img):
        cropped = img[int(img.shape[0] * 0.3)::,
                      int(img.shape[1]*0.2):int(img.shape[1] * 0.8):]
        # cropped = img[60::, ::]
        return cropped

    def __getitem__(self, idx):
        img = imread(self.files[idx])
        # img = self.crop_sky(img)
        img = resize(img, (196, 455)) * 255
        img = img.transpose(2, 0, 1)

        measurements = re.split('(?<!-)-', self.files[idx])
        angle = np.array([float(measurements[1])])

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(angle).float()

        return (img, label)
