import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from matplotlib.figure import Figure
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter, writer
from data import inf_data_gen, get_data

from config import cfg


_WRITER = SummaryWriter()


def _add_hparams(hparam_dict, metric_dict):
    exp, ssi, sei = writer.hparams(hparam_dict, {})
    _WRITER.file_writer.add_summary(exp)
    _WRITER.file_writer.add_summary(ssi)
    _WRITER.file_writer.add_summary(sei)


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


def train(net, X, T, optimizer, n_iter):
    P = net(X)
    loss = F.binary_cross_entropy(P, T)
    loss.backward()
    optimizer.step()
    _WRITER.add_scalar('Loss/train', loss.item(), n_iter)


def error(P, T):
    return 1 - torch.sum(torch.round(P) == T).float() / len(T)


def test(net, X, T, n_iter):
    with torch.no_grad():
        P = net(X)
        loss = F.binary_cross_entropy(P, T)
        _WRITER.add_scalar('Loss/test', loss.item(), n_iter)
        _WRITER.add_scalar('Error/test', error(P, T), n_iter)


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


def inv_root_lr(step):
    return 1 / np.sqrt(1 + step // 50_000)


def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lrs.append(param_group['lr'])
    assert len(set(lrs)) == 1
    return lrs[0]


if __name__ == '__main__':

    cfg.merge_from_list(['MODEL.W', int(sys.argv[1])])

    _add_hparams({**cfg.TRAIN, **cfg.MODEL, **cfg.DATA}, {})

    x_train, x_test, y_train, y_test = get_data()
    train_dataloader = inf_data_gen(x_train, y_train, cfg.TRAIN.BATCH_SIZE)
    X_test = torch.Tensor(x_test).to(cfg.SYSTEM.DEVICE)
    Y_test = torch.Tensor(y_test).to(cfg.SYSTEM.DEVICE)

    net = Net(D=cfg.MODEL.D, W=cfg.MODEL.W)
    net.to(cfg.SYSTEM.DEVICE)

    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    scheduler = LambdaLR(optimizer, lr_lambda=inv_root_lr)
    pbar = tqdm(train_dataloader, total=cfg.TRAIN.STEPS)
    for n_iter, (X, T) in enumerate(pbar, start=1):
        X, T = X.to(cfg.SYSTEM.DEVICE), T.to(cfg.SYSTEM.DEVICE)
        optimizer.zero_grad()
        net.train()
        train(net, X, T, optimizer, n_iter)

        if n_iter % 5000 == 0:
            net.eval()
            test(net, X_test, Y_test, n_iter)
            _WRITER.add_scalar('LR', get_lr(optimizer), n_iter)

        scheduler.step(n_iter)

        if n_iter % (cfg.TRAIN.STEPS / 10) == 0:
            analysis(net, x_train, y_train, n_iter)

        if n_iter > cfg.TRAIN.STEPS:
            break
