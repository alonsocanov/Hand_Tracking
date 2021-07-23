from torch.nn.modules.module import Module
from landmarks_dataset import Landmarks, Model
from torch.utils.data import random_split, DataLoader
import torch
from torch import nn
import torch.optim as optim
import argparse
from utils import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test", type=str,
                        help="choose [train, test]", default='train')
    parser.add_argument("--model_path", type=str,
                        help="Path to model", default='data/model.pth')
    args = parser.parse_args()

    dataset = Landmarks(path='data/landmarks.csv')

    dataset_len = len(dataset)
    print('Dataset lenght:', dataset_len)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using:', device)

    input_len = dataset.get_num_imputs()
    print('Input lenght:', input_len)

    labels = dataset.get_classes()
    num_labels = len(labels)
    print('Number of labels:', num_labels)
    print('Classes:', labels)

    train_len = int(.7 * dataset_len)
    test_len = dataset_len - train_len
    batch_size = 4
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_data = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=True, drop_last=True, num_workers=0)

    if args.train_test == 'train':
        model = Model(input_len, num_labels)
        model = model.double().to(device)
        print(model)
        # optimizations
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        epochs = 100

        model = train(epochs, train_data, model, optimizer,
                      criterion, device, batch_size, train_len)

        torch.save(model, args.model_path)

    elif args.train_test == 'test':
        # missing test on different dataset
        c_mtx = torch.zeros(num_labels, num_labels, dtype=torch.int64)
        total_correct = 0

        torch.no_grad()
        model = torch.load(args.model_path)
        # optimizations
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        model.eval()
        model.to(device)
        loss = 0
        total_samples = 0
        for i, instance in enumerate(test_data):
            x, y = instance
            x = x.squeeze()
            x.to(device)
            y.to(device)

            y_hat = model(x)
            loss += criterion(y_hat, y).item()

            y_max = torch.argmax(y_hat, dim=1)
            total_correct += sum(y_max == y)

            total_samples += batch_size
            for t, p, in zip(y.tolist(), y_max.tolist()):
                c_mtx[t, p] += 1

        print('Loss:', loss)
        print('Total predictions correct:', int(
            total_correct), 'of', total_samples)
        print('Acuracy:', float(total_correct / total_samples))
        print('Confusion Matrix:')
        print(c_mtx.numpy())


if __name__ == '__main__':
    main()
