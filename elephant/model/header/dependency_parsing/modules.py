import torch
from elephant.modules import MLP


class BiLinearAttention(torch.nn.Module):
    """
    Computes attention between two matrices using a bi-linear attention
    function. This function has a matrix of weights ``W`` and a bias ``b``, and
    the similarity between the two matrices ``X`` and ``Y`` is computed as
    ``X W Y^T + b``.

    Input:
        - mat1: ``(batch_size, num_rows_1, mat1_dim)``
        - mat2: ``(batch_size, num_rows_2, mat2_dim)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``
    """

    def __init__(self, mat1_dim: int, mat2_dim: int, use_input_biases: bool = False) -> None:
        super(BiLinearAttention, self).__init__()

        if use_input_biases:
            mat1_dim += 1
            mat2_dim += 1

        self.weight = torch.nn.Parameter(torch.zeros(1, mat1_dim, mat2_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self._use_input_biases = use_input_biases
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(0)

    def forward(self, mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
        if self._use_input_biases:
            bias1 = mat1.new_ones(mat1.size()[:-1] + (1,))
            bias2 = mat2.new_ones(mat2.size()[:-1] + (1,))

            mat1 = torch.cat([mat1, bias1], -1)
            mat2 = torch.cat([mat2, bias2], -1)

        intermediate = torch.matmul(mat1.unsqueeze(1), self.weight)
        final = torch.matmul(intermediate, mat2.unsqueeze(1).transpose(2, 3))
        return final.squeeze(1) + self.bias


class RelPredictor(torch.nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float = 0.1):
        super(RelPredictor, self).__init__()
        self.mlp = MLP(n_in, n_hidden, dropout=dropout, activation=True)
        self.linear = torch.nn.Linear(n_hidden, n_out)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.mlp(x)
        out = self.linear(out)
        return out


class LinearPredictor(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.1):
        super(LinearPredictor, self).__init__()
        self.linear = torch.nn.Linear(n_in, n_out)
        self.dropout = torch.nn.Dropout(p=dropout)

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.linear(self.dropout(x))
        return out
