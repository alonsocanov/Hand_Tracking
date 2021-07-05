import torch
import numpy as np


class Landmarks(Dataset):
    def __init__(self, path):

        self.x
        self.y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]
