"""
Defines a classifier
"""

import torch
from torch import nn
from torch.cuda import is_available

torch.set_default_dtype(torch.double)


class MLP(nn.Module):
    """
    Transformer based classifer to classify
    learning outcomes as skill or knowledge
    """

    def __init__(self, batch_size, dropout=0.4) -> None:

        super(MLP, self).__init__()

        self.device = 'cuda' if is_available() else 'cpu'
        self.dropout = dropout
        self.batch_size = batch_size

        self.MLP = nn.Sequential(
            nn.Linear(19, 256, device=self.device),
            nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, device=self.device),
            nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16, device=self.device),
            nn.Dropout(self.dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, device=self.device),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.MLP(x)
