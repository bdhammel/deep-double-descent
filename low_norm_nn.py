import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

N = 100
BATCHES = 100_000
BATCH_SZ = 32
D = 3
W = 100

writer = SummaryWriter(f'./logs/{datetime.now()}')


def twospirals(n_points, noise=.5):
    """Returns the two spirals dataset."""
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    d1x /= d1x.max()
    d1y /= d1y.max()
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


class Net(nn.Module):

    def __init__(self, D=1, W=100):
        super().__init__()
        self.D = D
        self.fc_in = nn.Linear(2, W)
        self.act_in = nn.Tanh()
        for l in range(1, D+1):
            setattr(self, f'fc_{l}', nn.Linear(W, W))
            setattr(self, f'act_{l}', nn.Tanh())
        self.fc_out = nn.Linear(W, 1)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.fc_in(x)
        x = self.act_in(x)
        for l in range(1, self.D+1):
            x = getattr(self, f'fc_{l}')(x)
            x = getattr(self, f'act_{l}')(x)
        x = self.fc_out(x)
        x = self.act_out(x)
        return x


def get_data():
    X, y = twospirals(1000)
    Y = y.reshape(-1, 1)
    return train_test_split(X, Y, test_size=0.33, random_state=42)


def inf_data_gen(X, y):
    all_idx = list(range(len(X)))
    while True:
        idx = np.random.choice(all_idx, replace=False, size=BATCH_SZ)
        x_batch = X[idx]
        y_batch = y[idx]
        yield torch.Tensor(x_batch), torch.Tensor(y_batch)


def train(net, X, T, optimizer, n_iter):
    L = net(X)
    loss = F.binary_cross_entropy(L, T)
    loss.backward()
    optimizer.step()
    writer.add_scalar('Loss/train', loss.item(), n_iter)


def test(net, X, T, n_iter):
    with torch.no_grad():
        L = net(X)
        loss = F.binary_cross_entropy(L, T)
        writer.add_scalar('Loss/test', loss.item(), n_iter)


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = get_data()
    train_dataloader = inf_data_gen(x_train, y_train)
    X_test = torch.Tensor(x_test)
    Y_test = torch.Tensor(y_test)

    net = Net(D=3)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    train_dataloader
    pbar = tqdm(train_dataloader, total=BATCHES)
    for n_iter, (X, T) in enumerate(pbar):
        optimizer.zero_grad()
        net.train()
        train(net, X, T, optimizer, n_iter)
        net.eval()
        test(net, X_test, Y_test, n_iter)
        if n_iter > BATCHES:
            break

    x = np.linspace(-1, 1, 50)
    XX, YY = np.meshgrid(x, x)
    env = np.c_[XX.ravel(), YY.ravel()]
    p = net(torch.Tensor(env)).detach().numpy().reshape(50, 50)
    plt.pcolormesh(XX, YY, p, cmap='bwr', vmin=0, vmax=1)
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test[:,0])
