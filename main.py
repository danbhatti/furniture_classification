import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms

import kaggle

import matplotlib.pyplot as plt

import os

def main():
    #kaggle.api.authenticate()

    #kaggle.api.dataset_download_files('udaysankarmukherjee/furniture-image-dataset', path='data/furniture', unzip=True)

    
    # classes of furniture
    f_classes = ['almirah', 'chair', 'fridge', 'table', 'tv']
    # associated directories with images for train/test
    folder_dirs = ["data/furniture/almirah_dataset", "data/furniture/chair_dataset", "data/furniture/fridge dataset",
                   "data/furniture/table dataset", "data/furniture/tv dataset"]
    # make dictionaries 
    class_dict = dict.fromkeys(f_classes)
    for class_instance in class_dict:
        class_dict[class_instance] = []
    # fill dictionary key (class) with values (images of that class)
    for i in range(len(folder_dirs)):
        for image in os.listdir(folder_dirs[i]):
            class_dict[f_classes[i]].append(f'{folder_dirs[i]}/{image}')
             
    train_data = []
    # make training data
    sample_size = 2000
    for class_instance in class_dict:
        for i in range(sample_size):
            test_example = [load_image(class_dict[class_instance][i]), class_instance]
            train_data.append(test_example)

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    furniture_dataset = CustomFurnitureDataset(train_data, transform=transform)
    trainloader = torch.utils.data.DataLoader(furniture_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    

def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()    

def load_image(image_path):
    """Loads an image from the given path, converts it to a NumPy array,
     and returns a torch.*Tensor object"""
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor_image = transform(image)
    return tensor_image

class CustomFurnitureDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




if __name__ == "__main__":
    main()