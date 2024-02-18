import torch
import elephant


class CustomTensorDataset(torch.utils.data.TensorDataset):
    @property
    def classes(self):
        return elephant.config.data_producer.processor.deprel_values


def dcov(X, Y):
    XY = torch.mul(X, Y)
    cov = torch.sqrt(XY.sum()) / X.shape[0]
    return cov


def dvar(X):
    return torch.sqrt(torch.sum(X ** 2 / X.shape[0] ** 2))


def cent_dist(X):
    M = torch.cdist(X, X, p=2)
    row_mean = M.mean(dim=1)
    column_mean = M.mean(dim=0)
    grand_mean = row_mean.mean()
    R = torch.tile(row_mean, (M.shape[0], 1)).transpose(0, 1)
    C = torch.tile(column_mean, (M.shape[1], 1))
    G = torch.tile(grand_mean, M.shape)
    CM = M - R - C + G
    return CM


def dcor(X, Y):
    assert X.shape[0] == Y.shape[0]
    A = cent_dist(X)
    B = cent_dist(Y)
    dcov_AB = dcov(A, B)
    dvar_A = dvar(A)
    dvar_B = dvar(B)

    dcor = 0.0
    if dvar_A > 0.0 and dvar_B > 0.0:
        dcor = dcov_AB / torch.sqrt(dvar_A * dvar_B)

    return dcor
