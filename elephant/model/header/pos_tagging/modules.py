import torch
import torch.nn as nn


class LinearPredictor(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.1):
        super(LinearPredictor, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.linear(self.dropout(x))
        return out
