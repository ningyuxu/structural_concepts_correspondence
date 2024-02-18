import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from functools import partial

from pathlib import Path

import elephant
from elephant.modules import MLP, RevGrad


class CKBFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hid_dim_1: int, hid_dim_2: int, dropout: float = 0.1):
        super(CKBFeatureEncoder, self).__init__()
        self.fc1 = MLP(input_dim, hid_dim_1, dropout=dropout)
        self.fc2 = MLP(hid_dim_1, hid_dim_2, dropout=0.0, activation_type="tanh")

    def forward(self, x):
        z = self.fc1(x)
        z = self.fc2(z)
        return z


class RelPredictor(nn.Module):
    def __init__(self, input_dim: int, num_cls: int, dropout: float = 0.1):
        super(RelPredictor, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(input_dim, num_cls)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        y = self.linear(self.dropout(x))
        return y


class DomainEncoder(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super(DomainEncoder, self).__init__()
        self.mlp = MLP(n_in, n_out, dropout=0, activation=False)

    def forward(self, x):
        out = self.mlp(x)
        return out


class FeatureEncoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.5):
        super(FeatureEncoder, self).__init__()
        self.mlp = MLP(n_in, n_out, dropout=dropout)

    def forward(self, x):
        out = self.mlp(x)
        return out


class FeatureDecoder(nn.Module):
    def __init__(self, n_in: int, n_out: int, dropout: float = 0.5):
        super(FeatureDecoder, self).__init__()
        self.mlp = MLP(n_in, n_out, dropout=dropout)

    def forward(self, x):
        out = self.mlp(x)
        return out


class DomainDiscriminator(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int = 2, dropout: float = 0.5):
        super(DomainDiscriminator, self).__init__()
        self.mlp = MLP(n_in, n_hidden, dropout=dropout, activation=True)
        self.linear = nn.Linear(n_hidden, n_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, constant):
        batch_size, seq_length, dim = x.size()
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, seq_length, -1)
        transposed = x.transpose(1, 2)
        diffs = x - transposed
        out = RevGrad.grad_reverse(diffs, constant)
        out = self.mlp(out)
        out = self.linear(out)
        return out


class StructureProbe(nn.Module):
    def __init__(self, n_in: int, rank: int = 64):
        super(StructureProbe, self).__init__()
        self.projector = nn.Parameter(data=torch.zeros(n_in, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.projector, -0.05, 0.05)

    def forward(self, x):
        transformed = torch.matmul(x, self.projector)
        batch_size, seq_length, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seq_length, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class RelationPredictor(nn.Module):
    def __init__(self, n_in: int, num_cls: int):
        super(RelationPredictor, self).__init__()
        self.linear = nn.Linear(n_in, num_cls)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        x = x.unsqueeze(2)
        x = x.expand(-1, -1, seq_length, -1)
        transposed = x.transpose(1, 2)
        diffs = x - transposed
        out = self.linear(diffs)
        return out


class L1DistanceLoss(nn.Module):
    """Custom L1 loss for distance matrices."""

    def __init__(self, device):
        super(L1DistanceLoss, self).__init__()
        self.device = device
        self.word_pair_dims = (1, 2)

    def forward(self, predictions, label_batch, length_batch):
        """
        Computes L1 loss on distance matrices.

        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.

        Args:
          predictions: A pytorch batch of predicted distances
          label_batch: A pytorch batch of true distances
          length_batch: A pytorch batch of sentence lengths

        Returns:
          A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss, total_sents


class UnionFind:
    """
    Naive UnionFind implementation for (slow) Prim's MST algorithm

    Used to compute minimum spanning trees for distance matrices
    """

    def __init__(self, n):
        self.parents = list(range(n))

    def union(self, i, j):
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i):
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


class LinearPhi(nn.Module):
    def __init__(self, dim):
        super(LinearPhi, self).__init__()
        self.dim = dim
        self.phi = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.phi.weight)
        nn.init.zeros_(self.phi.bias)

    def forward(self, x):
        out = self.phi(x)
        return out, out - x


class ResidualPhi(nn.Module):
    def __init__(self, n_blocks, dim, hid_dim=256, activation_type='relu',
                 norm_layer='none', n_branches=1, mapping_type="linear"):
        super(ResidualPhi, self).__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [ResBlock(dim, hid_dim, activation_type, norm_layer, n_branches, mapping_type) for _ in range(n_blocks)]
        )

    def forward(self, x):
        rs = []
        for block in self.blocks:
            x, r = block(x)
            rs.append(r)
        return x, rs

    def backward(self, y, max_iter=10):
        x = y
        for block in self.blocks:
            x = block.backward(x, max_iter=max_iter)
        return x


class ResidualMapping(nn.Module):
    def __init__(self, n_blocks, dim, hid_dim=256, activation_type='relu',
                 norm_layer='none', n_branches=1, dropout=0.0):
        super(ResidualMapping, self).__init__()
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList(
            [ResBlock(dim, hid_dim, activation_type, norm_layer, n_branches, mapping_type="bottleneck", dropout=dropout)
             for _ in range(n_blocks)]
        )

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x

    def backward(self, y, max_iter=10):
        x = y
        for block in self.blocks:
            x = block.backward(x, max_iter=max_iter)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, hid_dim, activation_type='relu', norm_layer='none',
                 n_branches=1, mapping_type="linear", dropout=0.0):
        super(ResBlock, self).__init__()
        self.coeff = 0.9
        self.n_power_iter = 1

        branches = []

        ##################
        # Linear Mapping #
        ##################
        if mapping_type == "linear":
            branches += [nn.Linear(int(dim), int(dim))]
            if norm_layer != 'none':
                branches += [self.get_norm_layer(norm_layer)(int(dim))]
            if activation_type != 'none':
                branches += [self.get_non_linearity(activation_type)()]

            for i in range(n_branches - 1):
                branches += [nn.Linear(int(dim), int(dim))]
                if norm_layer != 'none':
                    branches += [self.get_norm_layer(norm_layer)(int(dim))]
                if activation_type != 'none':
                    branches += [self.get_non_linearity(activation_type)()]

        ######################
        # Bottleneck Mapping #
        ######################
        else:
            branches += [nn.Linear(int(dim), int(hid_dim))]
            if norm_layer != 'none':
                branches += [self.get_norm_layer(norm_layer)(int(hid_dim))]
            if activation_type != 'none':
                branches += [self.get_non_linearity(activation_type)()]
            branches += [nn.Dropout(p=dropout)]
            branches += [nn.Linear(int(hid_dim), int(dim))]

            for i in range(n_branches - 1):
                if norm_layer != 'none':
                    branches += [self.get_norm_layer(norm_layer)(int(dim))]
                if activation_type != 'none':
                    branches += [self.get_non_linearity(activation_type)()]
                branches += [nn.Linear(int(dim), int(hid_dim))]
                if activation_type != 'none':
                    branches += [self.get_non_linearity(activation_type)()]
                branches += [nn.Dropout(p=dropout)]
                branches += [nn.Linear(int(hid_dim), int(dim))]

        self.branches = nn.Sequential(*branches)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.branches:
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                nn.init.orthogonal_(m.weight, gain=0.02)
                nn.init.zeros_(m.bias)
            if classname.find("Norm") != -1:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        r = self.branches(x)
        return x + r, r

    def backward(self, y, max_iter=10):
        x = y
        for iter_index in range(max_iter):
            summand = self.branches(x)
            x = y - summand
        return x

    @staticmethod
    def get_non_linearity(activation_type='relu'):
        if activation_type == 'relu':
            return partial(nn.ReLU, inplace=True)
        if activation_type == 'lrelu':
            return partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
        if activation_type == 'elu':
            return partial(nn.ELU, inplace=True)
        if activation_type == 'tanh':
            return nn.Tanh
        if activation_type == 'sigmoid':
            return partial(nn.Sigmoid)
        else:
            raise NotImplementedError('nonlinearity activitation [%s] is not found' % activation_type)

    @staticmethod
    def get_norm_layer(norm_type='none'):
        if norm_type == 'batch':
            norm_layer = partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'batch1d':
            norm_layer = partial(nn.BatchNorm1d, affine=True)
        elif norm_type == 'instance':
            norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'instance1d':
            norm_layer = partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
        elif norm_type == "layer":
            norm_layer = partial(nn.LayerNorm)
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer


class DomainClassifier(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int = 1, dropout: float = 0.5):
        super(DomainClassifier, self).__init__()
        self.mlp = MLP(n_in, n_hidden, dropout=dropout, activation=True)
        self.linear = nn.Linear(n_hidden, n_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.mlp(x)
        out = self.linear(out)
        return out


class Mapping(nn.Module):

    def __init__(self, input_dim: int, out_dim: int):
        super(Mapping, self).__init__()
        self.mapping = nn.Linear(input_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.mapping.weight)
        nn.init.zeros_(self.mapping.bias)

    def forward(self, x):
        return self.mapping(x)


class Phi(nn.Module):

    def __init__(
            self,
            input_dim: int = 768,
            hid_dim: int = 384,
            out_dim: int = 768,
            dropout: float = 0.33
    ):
        super(Phi, self).__init__()
        self.fc1 = MLP(input_dim, hid_dim, dropout=dropout)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# CKB loss
def CKBMetric(fea_s, fea_t, prob_s, plab_t, prob_t, num_cls, epsilon=1e-2, ckb_type='soft', device=elephant.device):
    #   Y: label, Z: fea, matching conditional distribution P(Z|Y)
    num_sam_s = fea_s.shape[0]
    num_sam_t = fea_t.shape[0]
    if ckb_type == 'hard':
        prob_t = torch.zeros(num_sam_t, num_cls).to(device).scatter(1, plab_t.unsqueeze(1), 1).detach()
    else:
        prob_t = prob_t.detach()
    I_s = torch.eye(num_sam_s).to(device)
    I_t = torch.eye(num_sam_t).to(device)

    # ====== Kernel Matrix and Centering Matrix =======
    H_s = (torch.eye(num_sam_s) - torch.ones(num_sam_s) / num_sam_s).to(device)
    H_t = (torch.eye(num_sam_t) - torch.ones(num_sam_t) / num_sam_t).to(device)

    D_YsYs = prob_s.pow(2).sum(1, keepdim=True).repeat(1, num_sam_s) + \
             prob_s.pow(2).sum(1, keepdim=True).t().repeat(num_sam_s, 1) - \
             2 * torch.mm(prob_s, prob_s.t())
    D_YtYt = prob_t.pow(2).sum(1, keepdim=True).repeat(1, num_sam_t) + \
             prob_t.pow(2).sum(1, keepdim=True).t().repeat(num_sam_t, 1) - \
             2 * torch.mm(prob_t, prob_t.t())
    D_ZsZs = fea_s.pow(2).sum(1, keepdim=True).repeat(1, num_sam_s) + \
             fea_s.pow(2).sum(1, keepdim=True).t().repeat(num_sam_s, 1) - \
             2 * torch.mm(fea_s, fea_s.t())
    D_ZtZt = fea_t.pow(2).sum(1, keepdim=True).repeat(1, num_sam_t) + \
             fea_t.pow(2).sum(1, keepdim=True).t().repeat(num_sam_t, 1) - \
             2 * torch.mm(fea_t, fea_t.t())
    D_ZtZs = fea_t.pow(2).sum(1, keepdim=True).repeat(1, num_sam_s) + \
             fea_s.pow(2).sum(1, keepdim=True).t().repeat(num_sam_t, 1) - \
             2 * torch.mm(fea_t, fea_s.t())

    sigma_YsYs = D_YsYs.mean().detach()
    sigma_YtYt = D_YtYt.mean().detach()
    sigma_ZsZs = D_ZsZs.mean().detach()
    sigma_ZtZt = D_ZtZt.mean().detach()
    sigma_ZtZs = D_ZtZs.mean().detach()

    K_YsYs = (-D_YsYs / sigma_YsYs).exp()
    K_YtYt = (-D_YtYt / sigma_YtYt).exp()
    K_ZsZs = (-D_ZsZs / sigma_ZsZs).exp()
    K_ZtZt = (-D_ZtZt / sigma_ZtZt).exp()
    K_ZtZs = (-D_ZtZs / sigma_ZtZs).exp()

    G_Ys = (H_s.mm(K_YsYs)).mm(H_s)
    G_Yt = (H_t.mm(K_YtYt)).mm(H_t)
    G_Zs = (H_s.mm(K_ZsZs)).mm(H_s)
    G_Zt = (H_t.mm(K_ZtZt)).mm(H_t)

    # ====== R_{s} and R_{t} =======
    Inv_s = (epsilon * num_sam_s * I_s + G_Ys).inverse()
    Inv_t = (epsilon * num_sam_t * I_t + G_Yt).inverse()
    R_s = epsilon * G_Zs.mm(Inv_s)
    R_t = epsilon * G_Zt.mm(Inv_t)

    # ====== R_{st} =======
    # B_s = I_s - (G_Ys - (G_Ys.mm(Inv_s)).mm(G_Ys))/(num_sam_s*epsilon)
    # B_t = I_t - (G_Yt - (G_Yt.mm(Inv_t)).mm(G_Yt))/(num_sam_t*epsilon)
    B_s = num_sam_s * epsilon * Inv_s
    B_t = num_sam_t * epsilon * Inv_t
    B_s = (B_s + B_s.t()) / 2  # numerical symmetrize
    B_t = (B_t + B_t.t()) / 2  # numerical symmetrize
    S_s, U_s = B_s.symeig(eigenvectors=True)
    S_t, U_t = B_t.symeig(eigenvectors=True)
    HC_s = H_s.mm(U_s.mm((S_s + 1e-4).pow(0.5).diag()))
    HC_t = H_t.mm(U_t.mm((S_t + 1e-4).pow(0.5).diag()))
    Nuclear = (HC_t.t().mm(K_ZtZs)).mm(HC_s)
    U_n, S_n, V_n = torch.svd(Nuclear)

    # ====== Conditional KB Distance
    CKB_dist = R_s.trace() + R_t.trace() - 2 * S_n[:-1].sum() / ((num_sam_s * num_sam_t) ** 0.5)

    return CKB_dist


# MMD loss
def MMDMetric(prob_s, prob_t):
    num_sam_s = prob_s.shape[0]
    num_sam_t = prob_t.shape[0]
    D_XsXs = prob_s.pow(2).sum(1, keepdim=True).repeat(1, num_sam_s) + \
             prob_s.pow(2).sum(1, keepdim=True).t().repeat(num_sam_s, 1) - \
             2 * torch.mm(prob_s, prob_s.t())
    D_XtXt = prob_t.pow(2).sum(1, keepdim=True).repeat(1, num_sam_t) + \
             prob_t.pow(2).sum(1, keepdim=True).t().repeat(num_sam_t, 1) - \
             2 * torch.mm(prob_t, prob_t.t())
    D_XtXs = prob_t.pow(2).sum(1, keepdim=True).repeat(1, num_sam_s) + \
             prob_s.pow(2).sum(1, keepdim=True).t().repeat(num_sam_t, 1) - \
             2 * torch.mm(prob_t, prob_s.t())

    sigma_XsXs = D_XsXs.mean().detach()
    sigma_XtXt = D_XtXt.mean().detach()
    sigma_XtXs = D_XtXs.mean().detach()

    K_XsXs = (-D_XsXs / sigma_XsXs).exp()
    K_XtXt = (-D_XtXt / sigma_XtXt).exp()
    K_XtXs = (-D_XtXs / sigma_XtXs).exp()

    MMD_dist = K_XsXs.mean() + K_XtXt.mean() - 2 * K_XtXs.mean()
    return MMD_dist


def mean_pooling(x):
    return torch.mean(x, dim=0, keepdim=True)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimplePrePoolNet(nn.Module):
    def __init__(self, input_dim, hid_dim, output_size=64, dropout=0.33):
        super(SimplePrePoolNet, self).__init__()
        self.output_size = output_size
        self.layer1 = MLP(input_dim, hid_dim, dropout=dropout)
        # self.layer2 = MLP(hid_dim_1, hid_dim_2, dropout=dropout)
        self.linear = nn.Linear(hid_dim, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.linear(x)
        x = x.view(x.size(0), -1)
        return x


class SetEncoder(nn.Module):
    def __init__(self, input_dim=768, hid_dim=384, output_size=64, dropout=0.33):
        super(SetEncoder, self).__init__()
        self.pre_pooling_fn = SimplePrePoolNet(
            input_dim=input_dim, hid_dim=hid_dim, output_size=output_size, dropout=dropout
        )
        self.pooling_fn = mean_pooling
        self.post_pooling_fn = Identity()

    def forward(self, x):
        """
        Forward pass through DeepSet SetEncoder. Implements the following computation:
                g(X) = rho ( mean ( phi(x) ) )
                Where X = (x0, ... xN) is a set of elements x in X (in our case, images from a context set)
                and the mean is a pooling operation over elements in the set.
        :param x: (torch.tensor) Set of elements X (e.g., for images has shape batch x C x H x W ).
        :return: (torch.tensor) Representation of the set, single vector in Rk.
        """
        x = self.pre_pooling_fn(x)
        x = self.pooling_fn(x)
        x = self.post_pooling_fn(x)
        return x


class LinearProj(nn.Module):
    def __init__(self, input_dim, rank):
        super(LinearProj, self).__init__()
        self.proj = nn.Parameter(data=torch.zeros(input_dim, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, x: torch.Tensor):
        x = torch.matmul(x, self.proj)
        return x

    def load(self, save_path: Path):
        assert save_path.is_file()
        to_load = torch.load(save_path).to(elephant.device)
        W = self.proj.data
        assert to_load.size() == W.size(), \
            f"Parameters loaded from {save_path} ({to_load.size()}) does not match the designated model ({W.size()})"
        W.copy_(to_load.type_as(W))
        # self.proj.to(elephant.device)

    def save(self, save_path: Path):
        W = self.proj.data.detach().clone().cpu()
        torch.save(W, save_path)


class AffineProj(nn.Module):
    def __init__(self, input_dim, output_dim, bias: bool = True):
        super(AffineProj, self).__init__()
        self.affine = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.affine.weight)
        if self.affine.bias is not None:
            nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor):
        x = self.affine(x)
        return x

    def save(self, save_path: Path):
        torch.save(self.affine.state_dict(), save_path)

    def load(self, save_path: Path):
        assert save_path.is_file(), f"Check adaptor weights stored at {save_path}"
        affine_state = torch.load(save_path, map_location="cpu")
        self.affine.cpu()
        self.affine.load_state_dict(affine_state)
        self.affine.to(elephant.device)


class Adaptor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = None, bias: bool = True, beta: float = 0.01):
        super(Adaptor, self).__init__()

        output_dim = input_dim if output_dim is None else output_dim
        self.beta = beta

        self.adaptor = nn.Linear(input_dim, output_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.eye_(self.adaptor.weight)
        # nn.init.xavier_normal_(self.adaptor.weight)
        if self.adaptor.bias is not None:
            nn.init.zeros_(self.adaptor.bias)

    def forward(self, x: torch.Tensor):  # , idx
        x = self.adaptor(x)
        # x = self.adaptors[idx](x)
        return x

    def orthogonalize(self):
        if self.beta > 0:
            W = self.adaptor.weight.data
            W.copy_((1 + self.beta) * W - self.beta * W.mm(W.transpose(0, 1).mm(W)))

    def save(self, save_path: Path):
        torch.save(self.adaptor.state_dict(), save_path)

    def load(self, save_path: Path):
        assert save_path.is_file(), f"Check adaptor weights stored at {save_path}"
        adaptor_state = torch.load(save_path, map_location="cpu")
        self.adaptor.cpu()
        self.adaptor.load_state_dict(adaptor_state)
        self.adaptor.to(elephant.device)


class LSAdaptor(nn.Module):
    def __init__(
            self,
            num_datasets: int,
            input_dim: int,
            output_dim: int = None,
            bias: bool = True,
            beta: float = 0.01
    ):
        super(LSAdaptor, self).__init__()

        output_dim = input_dim if output_dim is None else output_dim
        self.beta = beta
        self.num_datasets = num_datasets
        self.adaptors = nn.ModuleList()

        for i in range(self.num_datasets):
            self.adaptors.append(self._get_adaptor(size=output_dim, bias=bias))

    def _get_adaptor(self, size, bias):
        adaptor = nn.Linear(size, size, bias=bias)
        self.reset_parameters(adaptor)
        return adaptor

    @staticmethod
    def reset_parameters(m):
        # nn.init.eye_(self.adaptor.weight)
        nn.init.eye_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, idx):  # , idx
        x = self.adaptors[idx](x)
        return x

    def orthogonalize(self, idx):
        if self.beta > 0:
            W = self.adaptors[idx].weight.data
            W.copy_((1 + self.beta) * W - self.beta * W.mm(W.transpose(0, 1).mm(W)))

    def save(self, save_path: Path):
        torch.save(self.adaptors.state_dict(), save_path)

    def load(self, save_path: Path):
        assert save_path.is_file(), f"Check adaptor weights stored at {save_path}"
        adaptors_state = torch.load(save_path, map_location="cpu")
        self.adaptors.cpu()
        self.adaptors.load_state_dict(adaptors_state)
        self.adaptors.to(elephant.device)
