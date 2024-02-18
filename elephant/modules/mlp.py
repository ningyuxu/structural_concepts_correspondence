import torch

from .dropout import SharedDropout


class MLP(torch.nn.Module):
    """
    Applies a linear transformation together with a non-linear activation to the incoming tensor.
    """

    def __init__(self, n_in: int, n_out: int, dropout: float = .0, activation: bool = True, activation_type: str = ""):
        super(MLP, self).__init__()

        self.n_in = torch.tensor(n_in, dtype=torch.int)
        self.n_out = torch.tensor(n_out, dtype=torch.int)

        self.linear = torch.nn.Linear(n_in, n_out)

        if activation:
            if activation_type == "relu":
                self.activation = torch.nn.ReLU()
            elif activation_type == "elu":
                self.activation = torch.nn.ELU()
            elif activation_type == "tanh":
                self.activation = torch.nn.Tanh()
            elif activation_type == "sigmoid":
                self.activation = torch.nn.Sigmoid()
            else:
                self.activation = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            self.activation = torch.nn.Identity()

        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"
        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
