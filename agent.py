import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils import data 
#TODO: clean up these imports -- should NOT be importing something called data
from tqdm import tqdm
import random
import numpy as np
import numpy.random as np_rand
from user import ucb_user  # for use in ucb_recEngines
from torch.utils.data import DataLoader
from data import *

class recEngine(nn.Module):
    def __init__(self,
                 n_users=1,
                 n_items=1,
                 rank=1,
                 lamb=0.0001,
                 lr=0.01,
                 lr_decay=0.0001,
                 lr_min=0.01
                 ):
        super(recEngine, self).__init__()
        self.n_items = n_items
        self.n_users = n_users
        self.avg_r = {i: dict() for i in range(n_users)}
        self.n = {i: dict() for i in range(n_users)}
        #self.U = torch.sqrt(torch.ones(n_users,rank)/rank)
        #self.V = torch.sqrt(torch.ones(n_items,rank)/rank)
        self.U = torch.rand(n_users, rank)
        self.V = torch.rand(n_items, rank)
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.lamb = lamb
        self.data = []
        self.returns = 0
        self.total_recs = 0
        self.random_tally = 0

    def _update_avg_reward(self, user_id, item_id, interaction):
        if not item_id in self.avg_r[user_id]:
            self.avg_r[user_id][item_id] = 0
            self.n[user_id][item_id] = 0
        self.avg_r[user_id][item_id] *= self.n[user_id][item_id]
        self.avg_r[user_id][item_id] += interaction
        self.n[user_id][item_id] += 1
        self.avg_r[user_id][item_id] /= self.n[user_id][item_id]

    def _compute_lr(self, u, v):
        N_u = sum([self.n[u][i] for i in self.n[u]])
        N_v = sum([self.n[i][v] if v in self.n[i] else 0 for i in range(
            self.n_users)])

        def LR(N): return max(self.lr - N*self.lr_decay, self.lr_min)
        return LR(N_u), LR(N_v)

    def update(self, user_id, item_id, interaction):
        self.data.append((user_id, item_id, interaction))
        self.returns += interaction
        self.total_recs += 1
        u = int(user_id)
        v = int(item_id)
        self._update_avg_reward(u, v, interaction)
        lr_u, lr_v = self._compute_lr(u, v)
        def update(x, y, r, l): return (
            l*((r - torch.sum(x*y))*y - self.lamb*x))

        self.U[user_id] += update(
            self.U[user_id], self.V[item_id], self.avg_r[u][v], lr_u)
        self.V[item_id] += update(
            self.V[item_id], self.U[user_id], self.avg_r[u][v], lr_v)

    def _compute_prefs(self, user_id):
        return torch.sum(self.U[user_id] * self.V, dim=1)

    def rec(self, user_id):
        return torch.argmax(self._compute_prefs(user_id))

    def _seed_train(self):
        pass


class e_greedy_recEngine(recEngine):
    def __init__(self,
                 n_users=1,
                 n_items=1,
                 rank=1,
                 lamb=0.0001,
                 lr=0.01,
                 lr_decay=0.001,
                 lr_min=0.001,
                 eps=0.05,
                 eps_decay=0.000001,
                 eps_scaling=0.2
                 ):
        super().__init__(n_users=n_users, n_items=n_items, rank=rank,
                         lamb=lamb, lr=lr, lr_decay=lr_decay, lr_min=lr_min)
        self.eps = eps
        self.eps_decay = eps_decay
        self.tau = 1
        self.eps_scaling = eps_scaling

    def update(self, user_id, item_id, interaction):
        super().update(user_id, item_id, interaction)
        self.eps = 1/(self.tau**self.eps_scaling)
        self.tau += 1

    def rec(self, user_id):
        if random.random() > self.eps:
            item = super().rec(user_id)
        else:
            self.random_tally += 1
            item = random.randint(0, self.n_items-1)
        return item


class ucb_recEngine(recEngine):
    def __init__(self,
                 n_users=1,
                 n_items=1,
                 rank=1,
                 lamb=0.0001,
                 lr=0.01,
                 lr_decay=0.001,
                 lr_min=0.001,
                 ucb_scale=1
                 ):
        super().__init__(n_users=n_users, n_items=n_items, rank=rank,
                         lamb=lamb, lr=lr, lr_decay=lr_decay, lr_min=lr_min)
        self.ucb_bandit = ucb_user(self.n_items)
        self.ucb_bandit.lamb = ucb_scale

    def update(self, user_id, item_id, interaction):
        super().update(user_id, item_id, interaction)
        self.ucb_bandit.update(item_id, 0)

    def rec(self, user_id):
        ucb = self.ucb_bandit._ranking()
        return torch.argmax(self._compute_prefs(user_id) + ucb)


class ucb_fact_recEngine(recEngine):
    def __init__(self,
                 n_users=1,
                 n_items=1,
                 rank=1,
                 lamb=0.0001,
                 lr=0.01,
                 lr_decay=0.001,
                 lr_min=0.001,
                 ucb_scale=1,
                 alpha=0.1,
                 S=1,
                 L=1
                 ):
        super().__init__(n_users=n_users, n_items=n_items, rank=rank,
                         lamb=lamb, lr=lr, lr_decay=lr_decay, lr_min=lr_min)
        self.item_bandit = ucb_user(self.n_items)
        self.user_bandit = ucb_user(self.n_users)
        self.alpha = alpha
        self.S = S
        self.L = L
        self.d = rank
        self.V_bar = torch.stack(self.n_items*[alpha*torch.eye(rank)])
        self.U_bar = torch.stack(self.n_users*[alpha*torch.eye(rank)])

    def _confidence_ellipsoids(self, u, v):
        # CAUTION: make sure user and item vecs are col vecs
        self.U_bar[u] += torch.ger(self.U[u], self.U[u])
        self.V_bar[v] += torch.ger(self.V[v], self.V[v])

    def _ucb(self, t):
        return 0.5*torch.sqrt(self.d * torch.log(
            t + t**2*self.L**2/self.alpha)) + (self.alpha**0.5)*self.S

    def update(self, user_id, item_id, interaction):
        self._confidence_ellipsoids(user_id, item_id)
        super().update(user_id, item_id, interaction)
        self.item_bandit.update(item_id, 0)
        self.user_bandit.update(user_id, 0)

    def rec(self, user_id):
        item_ucb_list = self._ucb(self.item_bandit.n+1).numpy()
        user_ucb = self._ucb(self.user_bandit.n[user_id]+1).numpy()
        max_ip = -1*float('inf')
        best_item = -1
        for i, item_ucb in enumerate(item_ucb_list):
            _, _, ip, _ = self.AEIPM(
                self.V_bar[i].numpy(),
                self.U_bar[user_id].numpy(),
                item_ucb,
                user_ucb,
                self.V[i].numpy(),
                self.U[user_id].numpy())
            if ip > max_ip:
                best_item = i
                max_ip = ip
        assert best_item != -1, 'Error: No items found'
        return best_item

    def AEIPM(self, V, W, c, d, x, y, tol=0.001):
        x_k = x
        y_k = y
        eps = 1

        def update(alpha, beta, delta, M):
            # update alpha with beta fixed through M
            M_inv = np.linalg.inv(M)
            Ma = np.dot(M_inv, alpha)
            aMa = np.dot(alpha, Ma)
            lamb = np.sqrt(4*delta/aMa)
            return 0.5*lamb*Ma+beta

        while(eps > tol):
            y_prev = y_k
            x_prev = x_k
            y_k = update(x_k, y, d, V)
            x_k = update(x, y_k, c, W)
            ip = np.dot(x_k, y_k)
            eps = max(np.max(x_k - x_prev), np.max(y_k - y_prev))
        return x_k, y_k, ip, eps


class ss_ucb_recEngine(ucb_recEngine):
    def __init__(self,
                 n_users=1,
                 n_items=1,
                 rank=1,
                 lamb=0.0001,
                 lr=0.01,
                 lr_decay=0.001,
                 lr_min=0.001,
                 ss_rate=100,
                 init_setsize=1,
                 exploration_delay=True,
                 ucb_scale=1
                 ):
        super().__init__(n_users=n_users, n_items=n_items, rank=rank, lamb=lamb,
                         lr=lr, lr_decay=lr_decay, lr_min=lr_min, ucb_scale=ucb_scale)
        self.setmask = torch.zeros(n_items)
        self.t_offset = torch.zeros(n_items)
        self.r_perm = np_rand.permutation(range(n_items))
        self.setsize = 0
        self.xp_delay = exploration_delay
        self.ss_rate = ss_rate
        self._superset(init_setsize)

    def _superset(self, n_incr):
        assert self.setsize <= self.n_items, 'setsize should be <= n_items'
        if self.setsize < self.n_items:
            for i in range(n_incr):
                j = self.r_perm[i + self.setsize]
                self.setmask[j] += 1
                if self.xp_delay:
                    self.t_offset[j] += self.ucb_bandit.t
            self.setsize += n_incr

    def update(self, user_id, item_id, interaction):
        super().update(user_id, item_id, interaction)
        self.ucb_bandit.update(item_id, 0)
        if self.ucb_bandit.t % self.ss_rate == 0:
            self._superset(1)

    def rec(self, user_id):
        ucb = self.ucb_bandit.lamb*torch.sqrt(2*(torch.log(
            self.ucb_bandit.t-self.t_offset)/self.ucb_bandit.n))
        ranking = (self._compute_prefs(user_id) + ucb)*self.setmask
        return torch.argmax(ranking)


class SLAgent(nn.Module):
    def __init__(self, X=None, y=None, n_class=10):
        super(SLAgent, self).__init__()
        self.X = X
        self.y = y
        self.id = -1
        self.reward = 0
        self.wins = 0
        self.n_class = n_class  # set to zero for regression
        self.dataset_counts = torch.zeros(n_class)

    def add_data(self, x, y):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        if self.n_class > 0:  # set n_class = 0 for regression
            self.dataset_counts[y.long()] += 1

        if self.X is None:
            self.X = x
            self.y = y.float()
        else:
            self.X = torch.cat((self.X, x), dim=0)
            self.y = torch.cat((self.y, y.float()))

    def _update(self, x, y):
        pass

    def _seed_train(self):
        pass

    def predict(self, x):
        pass

    def get_reward(self, r):
        self.reward += r


class knnAgent(SLAgent):
    def __init__(self, X=None, y=None, k=1, eps=0.5, n_class=10):
        super(knnAgent, self).__init__(X=X, y=y, n_class=n_class)
        self.k = k
        self.eps = eps

    def target(self, x):
        return self._knn(x)[0] < self.eps

    def predict(self, x):
        knn = self.y[self._knn(x)[1]]
        val, cts = torch.unique(knn, return_counts=True, sorted=True)
        return val[torch.argmax(cts)]

    def get_reward(self, r):
        self.reward += r

    def bflat(self, T):
        return T.view(len(T), -1)

    def _knn(self, x):
        dist = torch.norm(self.bflat(self.X) - self.bflat(x), dim=1, p=None)
        return dist.topk(self.k, largest=False)


class onlineLLSR(SLAgent):
    # for now only supports univariate regressions
    def __init__(self, X=None, y=None, lamb=0, d=1):
        super(onlineLLSR, self).__init__(X=X, y=y, n_class=0)
        self.lamb = lamb  # for now ridge regression not supported
        assert lamb == 0, 'ridge regression not yet implemented'
        self.w = torch.zeros(d)
        self.H_inv = None

    def _seed_train(self):
        self.H_inv = (self.X @ self.X.t()).inverse()
        self.w = self.H_inv @ self.X @ self.y

    def _update(self, x, y):
        # apply Sherman-Morrison identity
        _x = x.squeeze()
        delta = self.H_inv @ torch.ger(_x, _x) @ self.H_inv
        self.H_inv -= (delta/(1 + x.t() @ self.H_inv @ x))
        self.w = self.H_inv @ self. X @ self.y
        
    def predict(self, x):
        return self.w.t() @ x


class Net(nn.Module):
    """Simple Network class."""

    def __init__(self, x_dim=28**2, n_class=10, hidden=512,
                 n_layers=2, task='C'):
        super(Net, self).__init__()
        self.x_dim = x_dim
        width = hidden if n_layers > 0 else n_class    
        self.layers = [nn.Linear(x_dim, width)]
        for i in range(n_layers): #num hidden layers
            self.layers.append(nn.Linear(width,width))
            self.layers.append(nn.LeakyReLU())
        if n_layers > 0:
            self.layers.append(nn.Linear(width,n_class))
            
        if task == 'C':
            self.layers.append(nn.Softmax(dim=1))
        elif task == 'R':
            raise ValueError('Regression not yet supported')
        else:
            raise ValueError('Not supported task type')

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        """Returns output after forward pass throug simple network."""
        x = x.view(x.shape[0], -1) #flatten
        return self.mlp(x.float())


class mlpAgent(SLAgent):

    def __init__(self, X=None, y=None, x_dim=28, n_class=10, n_layers=0,
                 hidden=512, task='C', epoch=10, lr=0.001, bsize=256,
                 retrain_limit=25, retrain_max=10**7):
        super(mlpAgent, self).__init__(X=X, y=y, n_class=n_class)
        # TODO : Fix n_class for regression tasks

        self.retrain_count = 0
        self.retrain_max = retrain_max
        self.retrain_limit = retrain_limit
        self.task = task
        self.epoch = epoch
        self.x_dim = x_dim
        self.network = Net(n_layers=n_layers, hidden=hidden, x_dim=x_dim)
        self.criterion = torch.nn.CrossEntropyLoss(size_average=False) if task == 'C' \
            else torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.accuracy = 0
        self.tested = 0
        self.batch_size = bsize

    def predict(self, x):
        """Returns the prediction on the inputs x."""
        output = self.network(x)
        _,output = torch.max(output, 1)
        return output

    def _update(self, x, y):

            
        self.network.train()
        self.optimizer.zero_grad()

        # Forward pass

        self.loss = self.criterion(self.network(x), y.long())
        self.loss.backward()
        self.optimizer.step()

        self.retrain_count += 1
        if self.retrain_count == self.retrain_limit:
            self._seed_train()
            self.retrain_count = 0

    def _seed_train(self):
        """Initialize training with seed data."""
        self.loss = 0.0
       
        dataset = data.TensorDataset(self.X,self.y)
        train_loader = data.DataLoader(dataset,
            batch_size = self.batch_size, shuffle = True)

        i = 0
        for epoch in range(self.epoch):
            for imgs,labels in iter(train_loader):
                self._update(imgs, labels)
                i += len(labels)
            y_pred = self.predict(self.X)
            

            acc = torch.mean((y_pred == self.y.squeeze()).float())
            print('epoch: ' + str(epoch) +  '; train acc: ' + str(acc.item()))

            if i >= self.retrain_max:
                break
           


class distance_sensitive_knnAgent(knnAgent):
    def __init__(self, X=None, y=None, k=1, eps=0.5, init_opt=1):
        super().__init__(X=X, y=y, k=k, eps=eps)
        self.dist_bucket_est = dict()
        self.init_opt = init_opt
        self.current_targ = None
        self.counter = dict()

    def target(self, x):
        d = self._knn(x)[0]
        i = int(d/self.eps)
        self.current_targ = i
        if not i in self.dist_bucket_est.keys():
            self.dist_bucket_est[i] = self.init_opt
            self.counter[i] = 1
        return self.dist_bucket_est[i] > 0

    def update(self, net_reward):
        update = self.counter[
            self.current_targ] * self.dist_bucket_est[self.current_targ]
        update += net_reward
        self.counter[self.current_targ] += 1
        update = update / self.counter[self.current_targ]
        self.dist_bucket_est[self.current_targ] = update
        self.current_targ = 0
