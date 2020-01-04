import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter, writer
from config import cfg


_WRITER = SummaryWriter()


def _add_hparams(hparam_dict, metric_dict):
    exp, ssi, sei = writer.hparams(hparam_dict, {})
    _WRITER.file_writer.add_summary(exp)
    _WRITER.file_writer.add_summary(ssi)
    _WRITER.file_writer.add_summary(sei)


def twospirals(n_points, noise=.5):
    """Returns the two spirals dataset"""
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    max_ = max(np.abs(d1x).max(), np.abs(d1y).max())
    d1x /= max_
    d1y /= max_
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_points), np.ones(n_points))))


class Net(nn.Module):

    def __init__(self, D, W):
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
    X, y = twospirals(cfg.DATA.N_POINTS, noise=cfg.DATA.NOISE)
    Y = y.reshape(-1, 1)
    return train_test_split(X, Y, test_size=0.91, random_state=42)


def inf_data_gen(X, y):
    all_idx = list(range(len(X)))
    while True:
        idx = np.random.choice(all_idx, replace=False, size=cfg.TRAIN.BATCH_SIZE)
        x_batch = X[idx]
        y_batch = y[idx]
        yield torch.Tensor(x_batch), torch.Tensor(y_batch)


def train(net, X, T, optimizer, n_iter):
    L = net(X)
    loss = F.binary_cross_entropy(L, T)
    loss.backward()
    optimizer.step()
    _WRITER.add_scalar('Loss/train', loss.item(), n_iter)


def test(net, X, T, n_iter):
    with torch.no_grad():
        L = net(X)
        loss = F.binary_cross_entropy(L, T)
        _WRITER.add_scalar('Loss/test', loss.item(), n_iter)


def analysis(net, x_train, y_train, n_iter=None):
    x = np.linspace(-2, 2, 150)
    XX, YY = np.meshgrid(x, x)
    X = torch.Tensor(np.c_[XX.ravel(), YY.ravel()]).to(cfg.SYSTEM.DEVICE)
    p = net(X).detach().cpu().numpy().reshape(150, 150)
    fig = Figure()
    ax = fig.subplots()
    ax.pcolormesh(XX, YY, p, cmap='bwr', vmin=0, vmax=1, alpha=.5)
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 0])
    _WRITER.add_figure('final_output', fig, n_iter)


if __name__ == '__main__':

    _add_hparams({**cfg.TRAIN, **cfg.MODEL, **cfg.DATA}, {})

    x_train, x_test, y_train, y_test = get_data()
    train_dataloader = inf_data_gen(x_train, y_train)
    X_test = torch.Tensor(x_test).to(cfg.SYSTEM.DEVICE)
    Y_test = torch.Tensor(y_test).to(cfg.SYSTEM.DEVICE)

    net = Net(D=cfg.MODEL.D, W=cfg.MODEL.W)
    net.to(cfg.SYSTEM.DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    pbar = tqdm(train_dataloader, total=cfg.TRAIN.STEPS)
    for n_iter, (X, T) in enumerate(pbar, start=1):
        X, T = X.to(cfg.SYSTEM.DEVICE), T.to(cfg.SYSTEM.DEVICE)
        optimizer.zero_grad()
        net.train()
        train(net, X, T, optimizer, n_iter)
        net.eval()
        test(net, X_test, Y_test, n_iter)

        if n_iter % (cfg.TRAIN.STEPS / 10) == 0:
            analysis(net, x_train, y_train, n_iter)

        if n_iter > cfg.TRAIN.STEPS:
            break
