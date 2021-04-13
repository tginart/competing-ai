import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as Transf
from torch.utils.data import Dataset


class linear_model(Dataset):
    def __init__(self, n_samp=10**8, d=1, gt_weights='ones', p_x='gaussian',
                 noise='gaussian', data_std=1, noise_std=0.2, bias=True):
        self.n_samp = n_samp
        if gt_weights == 'ones':
            self.w = torch.ones(d)
        elif gt_weights == 'gaussian':
            self.w = torch.randn(d)
        elif gt_weights == 'uniform':
            self.w = torch.rand(d)
        else:
            raise ValueError(f'Ground truth {gt_weights} is not supported')

        self.sample_x = self._sample(p_x, data_std, d)
        self.sample_eps = self._sample(noise, noise_std, 1)
        self.bias = bias

    def _sample(self, dist, sigma, d):
        if dist == 'uniform':
            def sample(): return torch.rand(d) - 0.5
        elif dist == 'gaussian':
            def sample(): return torch.randn(d)
        else:
            raise ValueError(f'Data distribution {dist} is not supported')
        return lambda: sigma*sample()

    def __len__(self):
        return self.n_samp

    def __getitem__(self, idx):
        x = self.sample_x()
        y = self.w @ x + self.sample_eps()
        if self.bias:
            _x = torch.zeros(len(x)+1)
            _x[:-1] = x
            _x[-1] += 1
            x = _x
        return (x, y)


class LowRankMatrix(Dataset):
    def __init__(self, n_users, n_items, rank_gt, n_samp):
        self.n_users = n_users
        self.n_items = n_items
        self.uEmb = torch.rand(n_users, rank_gt)
        self.vEmb = torch.rand(n_items, rank_gt)
        self.n_samp = n_samp
        self.sharpness = 1

    def __len__(self):
        return self.n_samp

    def __getitem__(self, idx):
        u = random.randint(0, self.n_users-1)
        v = random.randint(0, self.n_items-1)
        return (u, v)


def random_cluster_low_rank(n_users=1, n_items=1, rank_gt=1, n_samp=1,
                            pCTR=0.2, bonus=0.5):

    assert pCTR > 0.0, 'Require positive pCTR'
    assert pCTR < 1.0, 'Require pCTR less than 1'
    assert pCTR + bonus < 1.0, 'Require max pCTR less than 1'
    assert rank_gt >= 3, 'Need at least rank 3 for this dataset to work'

    # Note: some params are just used a method scope variables rather
    # than passed into the class -- consider fixing this at future time
    class ClusterLowRank(LowRankMatrix):

        def __init__(self, n_users, n_items, rank_gt, n_samp):
            super().__init__(n_users=n_users, n_items=n_items,
                             rank_gt=rank_gt, n_samp=n_samp)
            c = 0
            for v in range(n_items):
                self.vEmb[v] = torch.tensor([pCTR]*rank_gt)
                if c < rank_gt:
                    self.vEmb[v, c] += bonus
                c += 1

            for u in range(n_users):
                i = random.randint(0, rank_gt-1)
                self.uEmb[u] = torch.tensor([1.0/rank_gt]*rank_gt)
                self.uEmb[u, i] += bonus
                self.uEmb[u] /= torch.norm(self.uEmb[u])

    return ClusterLowRank(n_users, n_items, rank_gt, n_samp)


def random_pos_low_rank(n_users=1, n_items=1, rank_gt=1, n_samp=1):
    # low rank matrix completion
    return LowRankMatrix(n_users, n_items, rank_gt, n_samp)


def random_gaussian_low_rank(n_users=1, n_items=1, rank_gt=1, n_samp=1,
                             sharpness=1, var=1):
    # low rank matrix completion
    lrm = LowRankMatrix(n_users, n_items, rank_gt, n_samp)
    lrm.uEmb = (var**0.5)*torch.randn(lrm.uEmb.shape)
    lrm.vEmb = (var**0.5)*torch.randn(lrm.vEmb.shape)
    lrm.sharpness = sharpness
    return lrm


def random_minmax_gaussian_low_rank(n_users=1, n_items=1, rank_gt=1, n_samp=1,
                                    sharpness=1, var=1):

    def _minmax_scale(X, tol=1e-3):
        return (X - torch.min(X))/(torch.max(X) - torch.min(X) + tol)

    # low rank matrix completion
    lrm = LowRankMatrix(n_users, n_items, rank_gt, n_samp)
    uEmb = torch.randn(lrm.uEmb.shape)
    vEmb = torch.randn(lrm.vEmb.shape)
    lrm.sharpness = sharpness

    # scale the embeddings onto (0,1)
    M = _minmax_scale((var**0.5)*(uEmb @ vEmb.t()))
    # print(M.shape)
    # print(torch.max(M))
    # print(torch.min(M))

    # remake embeddings based on scaled ground truth
    U, S, V = torch.svd(M)
    lrm.uEmb = U[:, :rank_gt+1] @ torch.diag(S[:rank_gt+1]**0.5)
    lrm.vEmb = (torch.diag(S[:rank_gt+1]**0.5) @ V.t()[:rank_gt+1]).t()

    return lrm


def mnist(path='/home/'):
    return torchvision.datasets.MNIST(
        path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))]))


def fashion_mnist(path='/home/'):
    return torchvision.datasets.FashionMNIST(
        path,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor())


class healthDataset(Dataset):
    """Health Datasets Preprocessing."""

    def __init__(self, path = 'path goes here', 
                csv_prefix=None, transform=None, minmax = True, 
                tol = 1e-20):
        """
        Args: 
            path (string): datadir to find the csv files.
            csv_file (string): Prefix of the csv files, assuming it lives in 
                                datadir. To pass in 
                                Postures_X.csv, simply pass in 'Postures'
            transform (callable, optional): Optional transform to be applied
                on a sample.
            minmax (callable, optional): Optional minmax argument to transform
            the datasets, where each row is one data point. Notice that minmax
            is not for labels.  
            tol (callable, optional): Optional argument for tolerance in minmax
            scaling. 
        """
        csv_file_x = path + csv_prefix + '_X.csv'
        csv_file_y = path + csv_prefix + '_Y.csv'
        self.features = np.array(pd.read_csv(csv_file_x))
        self.labels = np.array(pd.read_csv(csv_file_y, header = None).iloc[:, 1])
        self.transform = transform
        self.minmax = minmax

        if self.minmax: 
            self.features = self._minmax_scale(self.features, tol)

    def __len__(self):
        return len(self.features)-1

    def _minmax_scale(self, X, tol):
        """ 
        Args: 
            X : a numpy array dataset to apply minmax scaling to. 
            tol: tolerance for numerical stability
        """
        numerator = X - np.min(X,axis=0)
        denominator = np.max(X,axis=0) - np.min(X,axis=0) + tol
        return numerator / denominator

    def __getitem__(self, idx):
        """ 
        Args:
            idx: scalar index at dataset (int). 
        Returns:
            A transformed sample for features and label.
            Features are shape (batch, n_features), elevation labels are 
            (batch,1).
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_features = self.features.shape[1]
        features = self.features[idx, :n_features]
        features = features.astype('float').reshape(-1)
        label = self.labels[idx].reshape(-1)
        sample = {'features': features, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample['features'], sample['label']
