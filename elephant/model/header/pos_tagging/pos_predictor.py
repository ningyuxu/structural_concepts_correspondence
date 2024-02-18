from typing import Dict, Tuple, Optional, Callable

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.nn import functional as F  # noqa

import elephant
from ..template import TaskModelTemplate
from .modules import LinearPredictor


class POSModel(TaskModelTemplate):
    POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}

    def __init__(self, task_cfg):
        super(POSModel, self).__init__(task_cfg)

        self.num_pos = len(self.task_cfg.upos_values)
        self.embedding_dim = self.task_cfg.embedding_dim

        self.pos_predictor = LinearPredictor(
            n_in=self.embedding_dim,
            n_out=self.num_pos,
            dropout=self.task_cfg.dropout
        )

        punctuation_tag_indices = {
            pos_tag: index
            for index, pos_tag in enumerate(self.task_cfg.upos_values) if pos_tag in self.POS_TO_IGNORE
        }
        self.pos_to_ignore = set(punctuation_tag_indices.values())

    def forward_loss(
            self,
            batch: Dict
    ) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)
        x = batch["embedding"]
        pos_tags = batch["postag_ids"]
        first_subword_mask = pos_tags != elephant.config.pad_label_id

        pos_tag_logits = self.pos_predictor(x)

        if torch.count_nonzero(first_subword_mask) > 0:  # noqa
            loss = criterion(pos_tag_logits.view(-1, self.num_pos), pos_tags.view(-1))
        else:
            loss = torch.zeros(1).squeeze(0).to(elephant.device)

        detail_loss = {"pos_loss": loss.item()}
        return loss, x.size(0), detail_loss

    def forward_loss_align(
            self,
            batch: Dict,
            proportion_t: torch.Tensor,
            proportion_s: torch.Tensor
    ) -> Tuple[torch.Tensor, int, Optional[Dict]]:

        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(proportion_t / proportion_s).to(elephant.device),
            ignore_index=elephant.config.pad_label_id
        ).to(elephant.device)

        x = batch["embedding"]
        pos_tags = batch["postag_ids"]
        first_subword_mask = pos_tags != elephant.config.pad_label_id
        pos_tag_logits = self.pos_predictor(x)

        if torch.count_nonzero(first_subword_mask) > 0:  # noqa
            loss = criterion(pos_tag_logits.view(-1, self.num_pos), pos_tags.view(-1))
        else:
            loss = torch.zeros(1).squeeze(0).to(elephant.device)

        detail_loss = {
            "weighted_pos_loss": loss.item()
        }
        return loss, x.size(0), detail_loss

    def forward_loss_ent(
            self,
            batch: Dict
    ) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        x = batch["embedding"]
        pos_tags = batch["postag_ids"]
        first_subword_mask = pos_tags != elephant.config.pad_head_id

        pos_tag_logits = self.pos_predictor(x)[first_subword_mask]

        ent_loss = self.entropy_loss(pos_tag_logits)

        if self.task_cfg.use_div:
            msoftmax = torch.nn.Softmax(dim=1)(pos_tag_logits).mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        else:
            div_loss = torch.zeros(1).squeeze(0).to(elephant.device)

        loss = self.task_cfg.ent_weight * ent_loss + self.task_cfg.div_weight * div_loss
        detail_loss = {"ent_loss": ent_loss.item(), "div_loss": div_loss.item()}

        return loss, x.size(0), detail_loss

    def extract_pos_pred(
            self,
            batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["embedding"]
        pos_tags = batch["postag_ids"]
        mask = pos_tags != elephant.config.pad_label_id

        pos_tag_logits = self.pos_predictor(x)

        pos_tag_logits = pos_tag_logits[mask].detach()
        pos_tags = pos_tags[mask]

        return pos_tag_logits, pos_tags

    def extract_pos_probs(
            self,
            batch: Dict,
            adapt: bool = False,
            adapter_func: Callable = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["embedding"]
        pos_tags = batch["postag_ids"]
        mask = pos_tags != elephant.config.pad_label_id

        pos_tag_logits = self.pos_predictor(x)
        pos_tag_probs = F.softmax(pos_tag_logits, dim=-1)

        if adapt:
            assert adapter_func is not None, "Check adapter function is provided"
            pos_tag_probs = adapter_func(pos_tag_probs)

        pos_tag_probs = pos_tag_probs[mask].detach()
        pos_tags = pos_tags[mask]
        return pos_tag_probs, pos_tags

    def evaluate(
            self,
            batch: Dict,
            adapt: bool = False,
            adapter_func: Callable = None
    ) -> Tuple[torch.Tensor, int, Dict, torch.Tensor, np.ndarray]:
        x = batch["embedding"]
        pos_tags = batch["postag_ids"]
        first_subword_mask = pos_tags != elephant.config.pad_label_id

        pos_tag_logits = self.pos_predictor(x)
        if adapt:
            assert adapter_func is not None, "Check adapter function is provided"
            pos_tag_probs = F.softmax(pos_tag_logits, dim=-1)
            adapted_probs = adapter_func(pos_tag_probs)
            predict_pos = adapted_probs.data.max(-1)[1]
            # pos_loss
            criterion = torch.nn.NLLLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)
            pos_loss = criterion(torch.log(adapted_probs.view(-1, self.num_pos) + 1e-4), pos_tags.view(-1))

        else:
            predict_pos = pos_tag_logits.data.max(-1)[1]
            # pos_loss
            criterion = torch.nn.CrossEntropyLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)
            pos_loss = criterion(pos_tag_logits.view(-1, self.num_pos), pos_tags.view(-1))

        if self.task_cfg.ignore_pos_punct:
            mask = self._get_mask_for_eval(first_subword_mask, pos_tags)  # noqa
        else:
            mask = first_subword_mask

        pos_tags = pos_tags.detach().cpu()
        predict_pos = predict_pos.detach().cpu()
        mask = mask.detach().cpu()

        pos_conf_mat = confusion_matrix(
            pos_tags[mask].numpy(), predict_pos[mask].long().numpy(), labels=np.arange(self.num_pos)
        )
        correct_labels = predict_pos.eq(pos_tags).long() * mask
        num_pos_correct = correct_labels.sum().item()

        loss = {"pos_loss": pos_loss.item()}

        return correct_labels, num_pos_correct, loss, mask, pos_conf_mat

    def evaluate_flow(
            self,
            x: torch.Tensor,
            y: torch.Tensor
    ) -> Tuple[torch.Tensor, int, Dict, torch.Tensor, np.ndarray]:
        pos_tags = y
        mask = pos_tags != elephant.config.pad_label_id

        criterion = torch.nn.CrossEntropyLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)

        pos_tag_logits = self.pos_predictor(x)

        pos_loss = criterion(pos_tag_logits.view(-1, self.num_pos), pos_tags.view(-1))
        predict_pos = pos_tag_logits.data.max(-1)[1]

        pos_tags = pos_tags.detach().cpu()
        predict_pos = predict_pos.detach().cpu()
        mask = mask.detach().cpu()

        pos_conf_mat = confusion_matrix(
            pos_tags[mask].numpy(), predict_pos[mask].long().numpy(), labels=np.arange(self.num_pos)
        )

        correct_labels = predict_pos.eq(pos_tags).long() * mask
        num_pos_correct = correct_labels.sum().item()

        loss = {"pos_loss": pos_loss.item()}

        return correct_labels, num_pos_correct, loss, mask, pos_conf_mat

    def _get_mask_for_eval(self, mask: torch.Tensor, pos_tags: torch.Tensor) -> torch.Tensor:
        """
        Dependency evaluation excludes words that are punctuation. Here, we create
        a new mask to exclude word indices which have a "punctuation-like"
        part of speech tag.

        Parameters
        ----------
        mask : `torch.BoolTensor`, required.
            The original mask.
        pos_tags : `torch.LongTensor`, required.
            The pos tags for the sequence.

        Returns
        -------
        A new mask, where any indices equal to labels we should be ignoring
        are masked.
        """
        new_mask = mask.detach()
        for label in self.pos_to_ignore:
            label_mask = pos_tags.eq(label)
            new_mask = new_mask & ~label_mask
        return new_mask.bool()

    @staticmethod
    def entropy_loss(v):
        """
        Entropy loss for probabilistic prediction vectors
        """
        return torch.mean(torch.sum(- F.softmax(v, dim=1) * F.log_softmax(v, dim=1), 1))
