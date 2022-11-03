import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch as th
from torch.optim import Adam, SGD
import numpy as np
from torch.nn import init


class Base_Exp_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.reward_history = []
        self.exp_reward_info = []


class RND_Model(Base_Exp_Model):
    def __init__(self, input_size, output_size, reward_scale=1):
        """
        input = observation
        output = any size
        """
        super().__init__(input_size, output_size)
        self.exp_model_key = 'RND'
        self.reward_scale = reward_scale

        self.predictor_net = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size),
        )
        self.target_net = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(self.input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size),
        )

        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight)
                init.uniform_(p.bias, -0.5, 0.5)

        for param in self.target_net.parameters():
            param.requires_grad = False

        # A weaker optimizer performs better than a strong one
        # self.optimizer = Adam(list(self.predictor_net.parameters()), lr=1e-4)
        self.optimizer = SGD(list(self.predictor_net.parameters()), lr=1e-4)
        self.optimizer_loss = 0
        self.optimizer_count = 0
        self.optimizer_batch_size = 64
        self.optimizer.zero_grad()

    def forward(self, obs):
        # self.optimizer.zero_grad()
        y = self.target_net(obs)
        y_hat = self.predictor_net(obs)

        loss = F.mse_loss(y_hat, y)
        self.optimizer_loss += loss
        self.optimizer_count += 1

        if self.optimizer_count == self.optimizer_batch_size:
            self.optimizer_loss.backward()
            self.optimizer.step()

            self.optimizer_count = 0
            self.optimizer_loss = 0
            self.optimizer.zero_grad()

        r_b = loss.detach().cpu().numpy().item()*self.reward_scale
        self.reward_history.append(r_b)

        return r_b

    def forward_batch_no_grad(self, grid: th.tensor):
        """
        grid: 1d-array of obs
        """
        with th.no_grad():
            y_hat = self.predictor_net(grid)
            y = self.target_net(grid)
            reward_grid = F.mse_loss(y_hat, y, reduction='none').sum(axis=1)
            reward_grid = reward_grid.detach().numpy()
            self.exp_reward_info.append(reward_grid)

        return None


class ACB_Model(Base_Exp_Model):
    def __init__(self, input_size, M, feature_type, reward_scale=1):
        """
        input: learned representation \phi(obs), can be last layer of value/policy net
        output: anti-concentration reward
        M: ensemble size / # of noisy_net
        """
        super().__init__(input_size, 1)
        self.exp_model_key = 'ACB'
        self.M = M
        self.feature_type = feature_type
        self.reward_scale = reward_scale

        self.noisy_nets = nn.ModuleList([nn.Linear(self.input_size, self.output_size) for i in range(self.M)])

        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         init.xavier_normal_(p.weight)
        #         init.uniform_(p.bias, -0.01, 0.01)

        # for p in self.modules():
        #     if isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()
        #         # init.uniform_(p.bias, -0.01, 0.01)

        self.optimizer = Adam(list(self.noisy_nets.parameters()), lr=1e-4)
        # self.optimizer = SGD(list(self.noisy_nets.parameters()), lr=1e-4)
        self.optimizer_loss = 0
        self.optimizer_count = 0
        self.optimizer_batch_size = 64
        self.optimizer.zero_grad()

        # list(model.noisy_nets.parameters())

    def forward(self, phi_obs):
        y_hat = th.empty(self.M)

        for i in range(self.M):
            y_hat[i] = self.noisy_nets[i](phi_obs)

        # generate random target
        y = th.randn(self.M)
        # y = torch.tensor(0.0).repeat(self.M)

        # optimize
        self.optimizer_loss += F.mse_loss(y_hat, y)
        self.optimizer_count += 1

        if self.optimizer_count == self.optimizer_batch_size:
            self.optimizer_loss.backward()
            self.optimizer.step()

            self.optimizer_count = 0
            self.optimizer_loss = 0
            self.optimizer.zero_grad()

        r_b = y_hat.detach().abs().max().numpy().item()*self.reward_scale
        self.reward_history.append(r_b)

        return r_b

    def forward_batch_no_grad(self, phi_grid: th.tensor, debug=True):
        with th.no_grad():
            y_hat = th.empty(len(phi_grid), self.M)

            for i in range(self.M):
                y_hat[:, i] = self.noisy_nets[i](phi_grid).squeeze()

            reward_grid = y_hat.max(axis=1).values.detach().numpy()
            self.exp_reward_info.append(reward_grid)

        if debug:
            print(reward_grid)

        return None

