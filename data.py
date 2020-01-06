import torch
import numpy as np
from sklearn.model_selection import train_test_split

from config import cfg


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


def make_data():
    X, y = twospirals(cfg.DATA.N_POINTS, noise=cfg.DATA.NOISE)
    Y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.90, random_state=42)
    np.save('./data/x_train', x_train)
    np.save('./data/y_train', y_train)
    np.save('./data/x_test', x_test)
    np.save('./data/y_test', y_test)


def get_data():
    x_train = np.load('./data/x_train.npy')
    y_train = np.load('./data/y_train.npy')
    x_test = np.load('./data/x_test.npy')
    y_test = np.load('./data/y_test.npy')
    return x_train, x_test, y_train, y_test


def inf_data_gen(X, y, batch_size):
    all_idx = list(range(len(X)))
    while True:
        idx = np.random.choice(all_idx, replace=False, size=batch_size)
        x_batch = X[idx]
        y_batch = y[idx]
        yield torch.Tensor(x_batch), torch.Tensor(y_batch)
