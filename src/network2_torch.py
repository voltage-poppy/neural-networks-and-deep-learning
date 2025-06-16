import gzip
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

def load_data() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return the MNIST dataset as a tuple of training, validation, and test data."""
    with gzip.open("../data/mnist.pkl.gz", 'rb') as f:
            training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data

class MnistDataset(Dataset):
    def __init__(self, data: tuple[np.ndarray, np.ndarray]):
        assert len(data[0]) == len(data[1]), "Number of features and labels must match"
        self.features = torch.tensor(data[0], dtype=torch.float32)
        self.labels = torch.tensor(data[1], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y


def load_data_wrapper() -> tuple[MnistDataset, MnistDataset, MnistDataset]:
    tr_d, va_d, te_d = load_data()

    training_data = MnistDataset(tr_d)
    validation_data = MnistDataset(va_d)
    test_data = MnistDataset(te_d)

    return training_data, validation_data, test_data


class MyNetwork(nn.Module):
    def __init__(self, sizes: list[int]):
        super(MyNetwork, self).__init__()
        self.sizes = sizes
        self.layers = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            nn.Sigmoid(),
            nn.Linear(sizes[1], sizes[2]),
            nn.Sigmoid()
        )

    def feedforward(self, x):
        return self.layers(x)

    def SGD(self, training_data_loader, epochs: int, eta: float, test_data_loader=None):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta)

        if test_data_loader is not None:
            num_test = len(test_data_loader)
        for j in range(epochs):
            for mini_batch in training_data_loader:
                x_batch, y_batch = mini_batch
                optimizer.zero_grad()
                outputs = self.feedforward(x_batch)
                # loss = F.mse_loss(outputs, y_batch)
                loss = F.cross_entropy(outputs, y_batch)
                loss.backward()
                optimizer.step()

            if test_data_loader is not None:
                acc = self.evaluate(test_data_loader)
                print(f"Epoch {j}: {acc} / {num_test}")
            else:
                print(f"Epoch {j} complete")

    def evaluate(self, test_data_loader) -> int:
        count = 0
        with torch.no_grad():
            for x, y in test_data_loader:
                output = self.feedforward(x)
                y_out = torch.argmax(output)
                if y_out.item() == y:
                    count += 1
        return count


if __name__ == "__main__":
    tr_d, va_d, te_d = load_data()
    training_ds = MnistDataset(tr_d)
    test_ds = MnistDataset(te_d)

    train_loader = DataLoader(training_ds, batch_size=10, shuffle=True)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    net = MyNetwork([784, 30, 10])
    num_epochs = 30
    learning_rate = 3.0
    net.SGD(train_loader, num_epochs, learning_rate, test_data_loader=test_loader)
