from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa
from torch.autograd import Variable
from torch.utils.data import Dataset as TorchDataset

import elephant
from elephant.utils.logging_utils import get_logger
from elephant.utils.torch_utils import freeze_module

from elephant.model.header.template import TaskModelTemplate
from elephant.model.header.syntax_aligning.modules import Mapping, Phi, Adaptor


logger = get_logger("elephant")


class MetaProtoNet(TaskModelTemplate):
    def __init__(self, task_cfg):
        super(MetaProtoNet, self).__init__(task_cfg)

        self.proto_net = Phi(
            input_dim=self.task_cfg.n_x, hid_dim=self.task_cfg.hid_dim, out_dim=self.task_cfg.proj_dim,
            dropout=self.task_cfg.dropout
        )
        self.uni_proj = Adaptor(
            input_dim=self.task_cfg.proj_dim, output_dim=self.task_cfg.proj_dim,
            bias=self.task_cfg.bias, beta=self.task_cfg.adaptor_beta,
        )

        self.all_labels = elephant.config.corpus.upos_values if self.task_cfg.task == "pos" \
            else elephant.config.corpus.deprel_values
        self.num_cls = len(self.all_labels)

    def freeze_module(self, freeze: bool = True):
        freeze_module(self.proto_net, freeze=freeze)
        freeze_module(self.uni_proj, freeze=freeze)

    def orthogonalize_uni_proj(self):
        self.uni_proj.orthogonalize()

    def embed(self, batch: Dict, task: str = "rel") -> Dict:
        if task == "pos":
            x = batch["embedding"]
            z = self.proto_net(x)
            batch["embedding"] = z
        else:
            x = batch["rel_representations"]
            z = self.proto_net(x)
            batch["rel_representations"] = z
        return batch

    def _embed_s_protos(self, x_protos: torch.Tensor):
        x = self.uni_proj(x_protos)
        return x

    def _get_features(self, samples: Dict):
        xs = samples["x_support"]
        if xs is not None:
            zs = self.proto_net(xs)
        else:
            zs = xs

        xq = samples["x_query"]
        zq = self.proto_net(xq)

        return zs, zq

    def forward_loss(self, samples: Dict) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        zs, zq = self._get_features(samples=samples)
        ys = samples["y_support"]
        yq = samples["y_query"]

        z_protos = samples["xs_proto"]
        z_protos = self.uni_proj(z_protos)
        y_protos = samples["ys_proto"]

        unique_classes = torch.unique(y_protos).detach().cpu().tolist()

        dists = self.euclidean_dist(zq, z_protos)

        log_p_y = F.log_softmax(-dists, dim=-1)
        loss_val, acc_val, _ = self._get_loss_acc_from_log_py(
            log_p_y=log_p_y, y_org=yq, unique_classes=unique_classes
        )

        detail_loss = {
            "loss_val": loss_val.item(),
            "acc_val": acc_val
        }
        return loss_val, zq.size(0), detail_loss

    def evaluate(self, samples: Dict) -> Tuple[Dict, Dict]:
        with torch.no_grad():
            z_protos = samples["xs_proto"]
            z_protos = self.uni_proj(z_protos)
            y_protos = samples["ys_proto"]
            ys = samples["y_support"]

            if self.task_cfg.task == "pos":
                xq_seq = samples["embedding"]
                yq_seq = samples["postag_ids"]
            else:
                xq_seq = samples["rel_representations"]
                yq_seq = samples["deprel_ids"]

            batch_size, seq_len = xq_seq.size(0), xq_seq.size(1)

            xq, yq = xq_seq.view(-1, xq_seq.size(-1)), yq_seq.view(-1)
            samples["x_query"] = xq
            samples["y_query"] = yq

            zs, zq = self._get_features(samples=samples)

            unique_classes = torch.unique(y_protos).detach().cpu().tolist()
            n_classes = len(unique_classes)

            dists = self.euclidean_dist(zq, z_protos)

            log_p_y = F.log_softmax(-dists, dim=-1)

            reindex_vals = range(0, n_classes)
            lmap = dict(zip(unique_classes, reindex_vals))
            lmap.update({elephant.config.pad_label_id: elephant.config.pad_label_id})

            num_ignored_valid_words = (torch.numel(yq) - torch.count_nonzero(
                torch.isin(yq, torch.LongTensor(unique_classes + [elephant.config.pad_label_id]).to(elephant.device))
            )).item()

            yq = torch.LongTensor([
                lmap[y.item()] if y.item() in lmap else elephant.config.pad_label_id for y in yq
            ]).to(elephant.device)

            target_inds = yq.unsqueeze(dim=1)
            range_vector = torch.arange(yq.size(0)).unsqueeze(1)

            mask = yq != elephant.config.pad_label_id
            loss_val = -log_p_y[range_vector, target_inds].squeeze().view(-1)[mask].mean()
            _, y_hat = log_p_y.max(1)

            y_hat = y_hat.view(batch_size, -1).detach()
            yq = yq.view(batch_size, -1).detach()
            mask = mask.view(batch_size, -1).detach()
            correct_labels = y_hat.eq(yq).long() * mask
            num_correct = correct_labels.sum().item()

            pred = y_hat[mask].detach().cpu().long().numpy()
            gold = yq[mask].detach().cpu().long().numpy()
            conf_mat = confusion_matrix(gold, pred, labels=np.arange(n_classes))

            metric = {
                "correct_labels": correct_labels.detach().cpu(),
                "num_correct": num_correct,
                "conf_mat": conf_mat,
                "mask": mask.detach().cpu(),
                "num_ignored_valid_words": num_ignored_valid_words
            }
            loss = {
                "loss_clf": loss_val.item(),
            }
        return metric, loss

    @staticmethod
    def _get_loss_acc_from_log_py(log_p_y, y_org, unique_classes):
        reindex_vals = range(0, len(unique_classes))
        lmap = dict(zip(unique_classes, reindex_vals))
        lmap.update({elephant.config.pad_label_id: elephant.config.pad_label_id})

        num_ignored_valid_words = (torch.numel(y_org) - torch.count_nonzero(
            torch.isin(y_org, torch.LongTensor(unique_classes + [elephant.config.pad_label_id]).to(elephant.device))
        )).item()

        y_adj = torch.LongTensor([
            lmap[y.item()] if y.item() in lmap else elephant.config.pad_label_id for y in y_org
        ]).to(elephant.device)
        target_inds = y_adj.unsqueeze(dim=1)
        range_vector = torch.arange(y_adj.size(0)).unsqueeze(1)

        mask = y_adj != elephant.config.pad_label_id
        assert torch.count_nonzero(mask) > 0, "Invalid sample, check training data"
        loss = -log_p_y[range_vector, target_inds].squeeze().view(-1)[mask].mean()
        _, y_hat = log_p_y.max(1)
        correct_labels = y_hat.eq(y_adj).long() * mask
        num_correct = correct_labels.sum().item()
        num_words = correct_labels.numel() - (1 - mask.long()).sum().item() + num_ignored_valid_words
        acc = num_correct / num_words
        detail_dict = {
            "correct_labels": correct_labels,
            "num_correct": num_correct,
            "mask": mask,
            "y_gold": y_adj,
            "y_pred": y_hat,
            "num_ignored_valid_words": num_ignored_valid_words,
        }
        return loss, acc, detail_dict

    @staticmethod
    def _get_protos(z_support: torch.Tensor, y_support: torch.Tensor):
        z_protos = []
        y_protos = []
        unique_classes = torch.unique(y_support).detach()
        for c in unique_classes:
            z_proto = z_support[y_support == c].mean(dim=-2, keepdim=True)
            z_protos.append(z_proto)
            y_protos.append(c)
        z_protos = torch.cat(z_protos, dim=0)
        y_protos = torch.LongTensor(y_protos).to(elephant.device)
        return z_protos, y_protos

    @staticmethod
    def euclidean_dist(x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    @staticmethod
    def label_to_onehot(label: torch.LongTensor, num_cls):
        return torch.zeros(label.shape[0], num_cls).to(elephant.device).scatter(1, label.unsqueeze(1), 1).detach()
