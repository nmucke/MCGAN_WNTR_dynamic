import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
import math
import torch.nn.init as init


def init_weights(m):
    if type(m) == nn.Linear:
        init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)


class Generator(nn.Module):
    def __init__(self, latent_dim=8,
                 par_dim=1,
                 hidden_channels=[],):

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.par_dim = par_dim
        self.hidden_channels = hidden_channels
        self.activation = nn.LeakyReLU()
        self.kernel_size = 5
        self.stride = 2
        self.sigmoid = nn.Sigmoid()

        #self.in_layer1 = nn.Linear(in_features=self.latent_dim,
        #                          out_features=self.hidden_channels[0])
        self.in_layer2 = nn.Linear(in_features=self.latent_dim,
                                  out_features=self.hidden_channels[0] * 4)

        self.batch_norm1 = nn.BatchNorm1d(self.hidden_channels[0])
        self.TransposedConv1 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[0],
                out_channels=self.hidden_channels[1],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.batch_norm2 = nn.BatchNorm1d(self.hidden_channels[1])
        self.TransposedConv2 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[1],
                out_channels=self.hidden_channels[2],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.batch_norm3 = nn.BatchNorm1d(self.hidden_channels[2])
        self.TransposedConv3 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[2],
                out_channels=self.hidden_channels[3],
                kernel_size=5,
                stride=self.stride,
                bias=True)
        self.batch_norm4 = nn.BatchNorm1d(self.hidden_channels[3])
        self.TransposedConv4 = nn.ConvTranspose1d(
                in_channels=self.hidden_channels[3],
                out_channels=self.hidden_channels[4],
                kernel_size=4,
                stride=2,
                bias=False)

    def forward(self, x):
        #x = self.activation(self.in_layer1(x))
        x = self.activation(self.in_layer2(x))
        x = x.view(x.size(0), self.hidden_channels[0], 4)
        x = self.batch_norm1(x)
        x = self.activation(self.TransposedConv1(x))
        x = self.batch_norm2(x)
        x = self.activation(self.TransposedConv2(x))
        x = self.batch_norm3(x)
        x = self.activation(self.TransposedConv3(x))
        x = self.batch_norm4(x)
        x = self.TransposedConv4(x)
        x = self.sigmoid(x)
        return x

class Critic(nn.Module):
    def __init__(self, hidden_channels=[]):
        super(Critic, self).__init__()

        self.hidden_channels = hidden_channels
        self.activation = nn.LeakyReLU()
        self.kernel_size = 5
        self.stride = 2


        self.Conv1 = nn.Conv1d(
                in_channels=self.hidden_channels[0],
                out_channels=self.hidden_channels[1],
                kernel_size=4,
                stride=2,
                bias=True)
        self.Conv2 = nn.Conv1d(
                in_channels=self.hidden_channels[1],
                out_channels=self.hidden_channels[2],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.Conv3 = nn.Conv1d(
                in_channels=self.hidden_channels[2],
                out_channels=self.hidden_channels[3],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)
        self.Conv4 = nn.Conv1d(
                in_channels=self.hidden_channels[3],
                out_channels=self.hidden_channels[4],
                kernel_size=self.kernel_size,
                stride=self.stride,
                bias=True)

        self.dense_layer = nn.Linear(in_features=self.hidden_channels[-1] * 4,
                                     out_features=self.hidden_channels[-1])

        self.out_layer = nn.Linear(in_features=self.hidden_channels[-1],
                                   out_features=1,
                                   bias=False)

    def forward(self, x, ):
        x = self.activation(self.Conv1(x))
        x = self.activation(self.Conv2(x))
        x = self.activation(self.Conv3(x))
        x = self.activation(self.Conv4(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.dense_layer(x))
        x = self.out_layer(x)
        return x