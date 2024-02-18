from typing import Dict, Tuple, Optional, Callable

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.nn import functional as F  # noqa

import elephant
from ..template import TaskModelTemplate
from .modules import RelPredictor, LinearPredictor


class RelModel(TaskModelTemplate):
    POS_TO_IGNORE = {"``", "''", ":", ",", ".", "PU", "PUNCT", "SYM"}

    def __init__(self, task_cfg):
        super(RelModel, self).__init__(task_cfg)

        self.num_rels = len(self.task_cfg.deprel_values)

        self.embedding_dim = self.task_cfg.embedding_dim
        if self.task_cfg.use_pos or self.task_cfg.use_predict_pos:
            self.embedding_dim += self.task_cfg.pos_dim

        if self.task_cfg.rel_predictor_type == "linear":
            self.rel_predictor = LinearPredictor(
                n_in=self.embedding_dim,
                n_out=self.num_rels,
                dropout=self.task_cfg.dropout
            )
        else:
            self.rel_predictor = RelPredictor(
                n_in=self.embedding_dim,
                n_hidden=self.task_cfg.hidden_dim,
                n_out=self.num_rels,
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
        x = batch["rel_representations"]
        head_tags = batch["deprel_ids"]
        first_subword_mask = head_tags != elephant.config.pad_label_id

        head_tag_logits = self.rel_predictor(x)

        if torch.count_nonzero(first_subword_mask) > 0:  # noqa
            loss = criterion(head_tag_logits.view(-1, self.num_rels), head_tags.view(-1))
        else:
            loss = torch.zeros(1).squeeze(0).to(elephant.device)

        detail_loss = {"rel_loss": loss.item()}

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

        x = batch["rel_representations"]
        head_tags = batch["deprel_ids"]
        first_subword_mask = head_tags != elephant.config.pad_label_id
        head_tag_logits = self.rel_predictor(x)

        if torch.count_nonzero(first_subword_mask) > 0:  # noqa
            loss = criterion(head_tag_logits.view(-1, self.num_rels), head_tags.view(-1))
        else:
            loss = torch.zeros(1).squeeze(0).to(elephant.device)

        loss *= self.task_cfg.t_dp_weight
        detail_loss = {
            "t_rel_loss": loss.item()
        }
        return loss, x.size(0), detail_loss

    def forward_loss_ent(
            self,
            batch: Dict
    ) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        x = batch["rel_representations"]
        head_ids = batch["head_ids"]
        first_subword_mask = head_ids != elephant.config.pad_head_id

        head_tag_logits = self.rel_predictor(x)[first_subword_mask]

        ent_loss = self.entropy_loss(head_tag_logits)

        if self.task_cfg.use_div:
            msoftmax = torch.nn.Softmax(dim=1)(head_tag_logits).mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        else:
            div_loss = torch.zeros(1).squeeze(0).to(elephant.device)

        loss = self.task_cfg.ent_weight * ent_loss + self.task_cfg.div_weight * div_loss
        detail_loss = {"ent_loss": ent_loss.item(), "div_loss": div_loss.item()}

        return loss, x.size(0), detail_loss

    def extract_rel_pred(
            self,
            batch: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["rel_representations"]
        head_tags = batch["deprel_ids"]
        mask = head_tags != elephant.config.pad_label_id

        head_tag_logits = self.rel_predictor(x)

        head_tag_logits = head_tag_logits[mask].detach()
        head_tags = head_tags[mask]

        return head_tag_logits, head_tags

    def extract_rel_label_probs(
            self,
            batch: Dict,
            adapt: bool = False,
            adapter_func: Callable = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = batch["rel_representations"]
        head_tags = batch["deprel_ids"]
        mask = head_tags != elephant.config.pad_label_id

        head_tag_logits = self.rel_predictor(x)
        head_tag_probs = F.softmax(head_tag_logits, dim=-1)

        if adapt:
            assert adapter_func is not None, "Check adapter function is provided"
            head_tag_probs = adapter_func(head_tag_probs)

        head_tag_probs = head_tag_probs[mask].detach()
        head_tags = head_tags[mask]
        return head_tag_probs, head_tags

    def evaluate(
            self,
            batch: Dict,
            adapt: bool = False,
            adapter_func: Callable = None
    ) -> Tuple[torch.Tensor, int, Dict, torch.Tensor, np.ndarray]:
        x = batch["rel_representations"]
        head_tags = batch["deprel_ids"]
        first_subword_mask = head_tags != elephant.config.pad_label_id

        head_tag_logits = self.rel_predictor(x)
        if adapt:
            assert adapter_func is not None, "Check adapter function is provided"
            head_tag_probs = F.softmax(head_tag_logits, dim=-1)
            adapted_probs = adapter_func(head_tag_probs)
            predict_rels = adapted_probs.data.max(-1)[1]
            # rel_loss
            criterion = torch.nn.NLLLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)
            rel_loss = criterion(torch.log(adapted_probs.view(-1, self.num_rels) + 1e-4), head_tags.view(-1))

        else:
            predict_rels = head_tag_logits.data.max(-1)[1]
            # rel_loss
            criterion = torch.nn.CrossEntropyLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)
            rel_loss = criterion(head_tag_logits.view(-1, self.num_rels), head_tags.view(-1))

        if self.task_cfg.ignore_pos_punct:
            postag_ids = batch["postag_ids"]
            mask = self._get_mask_for_eval(first_subword_mask, postag_ids)  # noqa
        else:
            mask = first_subword_mask

        head_tags = head_tags.detach().cpu()
        predict_rels = predict_rels.detach().cpu()
        mask = mask.detach().cpu()

        rel_conf_mat = confusion_matrix(
            head_tags[mask].numpy(), predict_rels[mask].long().numpy(), labels=np.arange(self.num_rels)
        )

        correct_labels = predict_rels.eq(head_tags).long() * mask
        num_rel_correct = correct_labels.sum().item()

        loss = {"rel_loss": rel_loss.item()}

        return correct_labels, num_rel_correct, loss, mask, rel_conf_mat

    def evaluate_flow(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, int, Dict, torch.Tensor, np.ndarray]:
        head_tags = y
        mask = head_tags != elephant.config.pad_label_id

        criterion = torch.nn.CrossEntropyLoss(ignore_index=elephant.config.pad_label_id).to(elephant.device)

        head_tag_logits = self.rel_predictor(x)

        rel_loss = criterion(head_tag_logits.view(-1, self.num_rels), head_tags.view(-1))
        predict_rels = head_tag_logits.data.max(-1)[1]

        head_tags = head_tags.detach().cpu()
        predict_rels = predict_rels.detach().cpu()
        mask = mask.detach().cpu()

        rel_conf_mat = confusion_matrix(
            head_tags[mask].numpy(), predict_rels[mask].long().numpy(), labels=np.arange(self.num_rels)
        )

        correct_labels = predict_rels.eq(head_tags).long() * mask
        num_rel_correct = correct_labels.sum().item()

        loss = {"rel_loss": rel_loss.item()}

        return correct_labels, num_rel_correct, loss, mask, rel_conf_mat

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
