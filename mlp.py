import torch
import torch.nn as nn
import random

class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU, dropout=0.0):
        super(MLP, self).__init__()
        self.weight = nn.ParameterList([torch.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.b = nn.ParameterList([torch.randn(layers[i+1]) for i in range(len(layers)-1)])
        self.activation = activation()
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.weight)):
            if self.dropout > 0 and self.training:
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)                 
            x = x @ self.weight[i] + self.b[i]
            if i != len(self.weight) - 1:
                x = self.activation(x)
        return x
    
