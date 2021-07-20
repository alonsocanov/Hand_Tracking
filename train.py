from torch.nn.modules.module import Module
from landmarks_dataset import Landmarks, Model
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn
import torch.optim as optim

import argparse


def train(epochs, train_data, model, optimizer, criterion, device, batch_size, train_lenght):

    for epoch in range(1, epochs + 1):
        for i, instance in enumerate(train_data):
            x, y = instance
            x = x.squeeze()
            x.to(device)
            y.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(x)
            # calculate loss
            loss = criterion(yhat, y)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            if not (i + 1) % 2:
                print('Epoch %d, Sample: %5d/%5d Loss: %.3f' %
                      (epoch, (i + 1) * batch_size, train_lenght, loss))
    return model


def main():
    dataset = Landmarks(path='data/landmarks.csv')

    dataset_len = len(dataset)
    print('Dataset lenght:', dataset_len)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using:', device)

    input_len = dataset.get_num_imputs()
    print('Input lenght:', input_len)

    labels = dataset.get_classes()
    print('Classes:', labels)
    num_labels = len(labels)
    print('Number of labels:', num_labels)

    train_len = int(.7 * dataset_len)
    test_len = dataset_len - train_len
    batch_size = 4
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_data = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=True, drop_last=True, num_workers=0)

    model = Model(input_len, num_labels)
    model = model.double().to(device)
    print(model)

    # optimizations
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    epochs = 100

    model = train(epochs, train_data, model, optimizer,
                  criterion, device, batch_size, train_len)

    model_path = 'data/model.pth'
    torch.save(model, model_path)


if __name__ == '__main__':
    main()
