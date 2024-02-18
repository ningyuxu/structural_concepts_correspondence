from typing import Dict, Tuple, Optional
from pathlib import Path

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
from elephant.model.header.syntax_aligning.modules import Mapping, Phi, SetEncoder, LinearProj
from elephant.model.utils import dcor as compute_dcor


logger = get_logger("elephant")


class ProbeModel(TaskModelTemplate):
    def __init__(self, task_cfg):
        super(ProbeModel, self).__init__(task_cfg)

        self.proj = LinearProj(input_dim=self.task_cfg.n_x, rank=self.task_cfg.proj_dim)
        self.all_labels = elephant.config.corpus.upos_values if self.task_cfg.task == "pos" \
            else elephant.config.corpus.deprel_values
        self.num_cls = len(self.all_labels)

    def freeze_module(self, freeze: bool = True):
        freeze_module(self.proj, freeze=freeze)

    def save(self, save_path: Path):
        self.proj.save(save_path=save_path)

    def load(self, save_path: Path):
        self.proj.load(save_path=save_path)

    def embed(self, batch: Dict) -> Dict:
        if self.task_cfg.task == "pos":
            x = batch["embedding"]
            z = self.proj(x)
            batch["embedding"] = z
        else:
            x = batch["rel_representations"]
            z = self.proj(x)
            batch["rel_representations"] = z
        return batch

    def forward_loss(self, batch: Dict, protos: Dict) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        x_protos = protos["xs_proto"]
        y_protos = protos["ys_proto"]

        if self.task_cfg.task == "pos":
            xq_seq = batch["embedding"]
            yq_seq = batch["postag_ids"]
        else:
            xq_seq = batch["rel_representations"]
            yq_seq = batch["deprel_ids"]

        batch_size, seq_len = xq_seq.size(0), xq_seq.size(1)

        xq, yq = xq_seq.view(-1, xq_seq.size(-1)), yq_seq.view(-1)
        unique_classes = torch.unique(y_protos).detach().cpu().tolist()

        z_protos = self.proj(x_protos)
        zq = self.proj(xq)

        dists = self.euclidean_dist(zq, z_protos)

        log_p_y = F.log_softmax(-dists, dim=-1)

        reindex_vals = range(0, len(unique_classes))
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
        # mask = yq_seq != elephant.config.pad_label_id
        assert torch.count_nonzero(mask) > 0, "Invalid sample, check training data"
        loss_val = -log_p_y[range_vector, target_inds].squeeze().view(-1)[mask].mean()
        _, y_hat = log_p_y.max(1)

        y_hat = y_hat.view(batch_size, -1).detach()
        yq = yq.view(batch_size, -1).detach()
        mask = mask.view(batch_size, -1).detach()
        correct_labels = y_hat.eq(yq).long() * mask
        num_correct = correct_labels.sum().item()
        num_words = correct_labels.numel() - (1 - mask.long()).sum().item() + num_ignored_valid_words
        acc_val = num_correct / num_words

        detail_loss = {
            "loss_val": loss_val.item(),
            "acc_val": acc_val
        }
        return loss_val, zq.size(0), detail_loss

    def evaluate(self, batch: Dict, protos: Dict) -> Tuple[Dict, Dict]:
        with torch.no_grad():
            x_protos = protos["xs_proto"]
            y_protos = protos["ys_proto"]
            if self.task_cfg.task == "pos":
                xq_seq = batch["embedding"]
                yq_seq = batch["postag_ids"]
            else:
                xq_seq = batch["rel_representations"]
                yq_seq = batch["deprel_ids"]

            batch_size, seq_len = xq_seq.size(0), xq_seq.size(1)

            xq, yq = xq_seq.view(-1, xq_seq.size(-1)), yq_seq.view(-1)
            unique_classes = torch.unique(y_protos).detach().cpu().tolist()
            n_classes = len(unique_classes)

            z_protos = self.proj(x_protos)
            zq = self.proj(xq)

            dists = self.euclidean_dist(zq, z_protos)

            log_p_y = F.log_softmax(-dists, dim=-1)

            num_ignored_valid_words = (torch.numel(yq) - torch.count_nonzero(
                torch.isin(yq, torch.LongTensor(unique_classes + [-1]).to(elephant.device))
            )).item()

            # get mapped labels
            reindex_vals = range(0, len(unique_classes))
            lmap = dict(zip(unique_classes, reindex_vals))
            lmap.update({elephant.config.pad_label_id: elephant.config.pad_label_id})

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
                "num_ignored_valid_words": num_ignored_valid_words,
            }
            loss = {
                "loss_clf": loss_val.item(),
            }
        return metric, loss

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
