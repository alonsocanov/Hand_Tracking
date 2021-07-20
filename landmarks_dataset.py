import torch
from torch.utils.data import Dataset
from torch.nn import Module, Linear, Sigmoid, ReLU, Sequential, functional
import numpy as np
import pandas as pd


class Landmarks(Dataset):
    def __init__(self, path):

        self.data = pd.read_csv(path)
        # # removed distance 0 and 17
        # self.x = data[['1', '2', '3', '4', '5', '6', '7', '8', '9',
        #                '10', '11', '12', '13', '14', '15', '16', '18', '19', '20']]
        # self.y = data['label']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        distance = np.asarray(self.data.iloc[idx, 1:])
        distance = distance.astype(float).reshape(1, -1)
        distance = torch.from_numpy(distance)
        label = np.asarray(self.data.iloc[idx, 0])
        label = label.astype(int)
        label = torch.from_numpy(label)
        return (distance, label)

    def get_classes(self):
        return self.data.iloc[:, 0].unique()

    def get_num_imputs(self):
        return self.data.shape[1] - 1


class Model(Module):
    # define model elements
    def __init__(self, n_inputs, n_classes):
        super().__init__()
        self.sigmoid = Sigmoid()
        self.relu = ReLU()
        self.l1 = Linear(n_inputs, 128)
        self.l2 = Linear(128, 512)
        self.l3 = Linear(512, 128)
        self.l4 = Linear(128, n_classes)

    # forward propagate input
    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.l4(x)
        return x
