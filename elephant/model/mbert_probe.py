from tqdm import tqdm
from typing import Tuple, Dict, Callable
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

import elephant
from elephant.data import DataLoader
from elephant.trainer.result import Result
from elephant.utils.logging_utils import get_logger

from elephant.model.encoder import get_encoder
from .header import get_task
from .template import ModelTemplate
from .utils import CustomTensorDataset


logger = get_logger("elephant")


class MBertProbeModel(ModelTemplate):
    def __init__(self, model_cfg):
        super(MBertProbeModel, self).__init__(model_cfg)
        self.encoder = get_encoder(self.model_cfg.encoder)
        self.relation_predictor = get_task(self.model_cfg.task.rel_module)
        self.arc_biaffine = get_task(self.model_cfg.task.arc_module)
        self.probe_module = get_task(self.model_cfg.task.probe_module)
        # self.align_module = get_task(self.model_cfg.task.align_module)

    def activate_adapter(self, adapter_name: str):
        self.encoder.activate_adapter(adapter_name)

    def freeze_adapter(self, adapter_name: str):
        self.encoder.freeze_adapter(adapter_name)

    def freeze_probe_module(self, freeze: bool = True):
        self.probe_module.freeze_module(freeze)

    def save_probe(self, save_path: Path):
        self.probe_module.save(save_path=save_path)

    def load_probe(self, save_path: Path):
        self.probe_module.load(save_path=save_path)

    def embed_data(self, batch: Dict) -> Dict:
        """
        Call encoder to embed data
        """
        batch = self.encoder.embed(batch)
        return batch

    def forward_loss(self, batch) -> Tuple[Result, int]:
        batch = self.encoder.embed(batch)
        s_loss_arc, count, s_detail_loss_arc = self.arc_biaffine.forward_loss(batch=batch)
        s_detail_loss_dp = {
            "s_pos_loss": s_detail_loss_arc["pos_loss"],
            "s_arc_loss": s_detail_loss_arc["arc_loss"],
        }
        batch = self.embed_rel(batch=batch)
        loss, count, detail_loss = self.relation_predictor.forward_loss(batch=batch)
        loss += s_loss_arc
        detail_loss.update(s_detail_loss_dp)
        log_header = "{:<20} | {:<20} | {:<20}".format(
            "total_loss", "s_arc_loss", "s_rel_loss"
        )
        log_line = "{:<20} | {:<20} | {:<20}".format(
            loss.item(),
            detail_loss["s_arc_loss"],
            detail_loss["rel_loss"]
        )
        loss_result = Result(
            metric_score=loss,
            log_header=log_header,
            log_line=log_line,
            metric_detail=detail_loss,
        )
        return loss_result, count

    def forward_loss_probe(self, batch: Dict, protos_dict: Dict):
        batch = self.encoder.embed(batch)
        batch = self.embed_rel(batch=batch)
        loss, count, detail_loss = self.probe_module.forward_loss(batch=batch, protos=protos_dict)

        log_header = "{:<20} | {:<20}".format("loss_val", "acc_val")
        log_line = "{:<20} | {:<20}".format(
            detail_loss["loss_val"], detail_loss["acc_val"]
        )

        loss_result = Result(
            metric_score=loss,
            log_header=log_header,
            log_line=log_line,
            metric_detail=detail_loss,
        )
        return loss_result, count

    # --------------------------------------------------------------------------------------------
    # evaluate model
    # --------------------------------------------------------------------------------------------
    def evaluate(
            self,
            dataloader: DataLoader,
            domain: str = "source",
            adapt: bool = False,
            adapter_func: Callable = None
    ) -> Tuple[Result, Result]:
        unlabeled_correct = 0.0
        full_unlabeled_correct = 0.0
        labeled_correct = 0.0
        full_labeled_correct = 0.0
        rel_correct = 0.0

        rel_confusion_matrix = np.zeros((self.relation_predictor.num_rels, self.relation_predictor.num_rels))

        total_sentences = 0.0
        total_words = 0.0

        pos_loss = 0.0
        arc_loss = 0.0
        rel_loss = 0.0

        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        unlabeled_full_match = 0.0
        labeled_full_match = 0.0

        with torch.no_grad():
            seen_batch = 0
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)
                metric, loss, correct_heads, mask = self.arc_biaffine.evaluate(batch=batch)  # noqa
                batch_rel = self.embed_rel(batch=batch)

                correct_labels, num_rel_correct, rel_loss_dict, mask, rel_conf_mat = \
                    self.relation_predictor.evaluate(batch=batch_rel,  adapt=adapt, adapter_func=adapter_func)  # noqa

                correct_labels_and_indices = correct_heads * correct_labels
                batch_labeled_full_match = (correct_labels_and_indices + (1 - mask.long())).prod(dim=-1)
                batch_labeled_correct = correct_labels_and_indices.sum().item()
                batch_full_labeled_correct = batch_labeled_full_match.sum().item()
                metric["labeled_correct"] = batch_labeled_correct
                metric["full_labeled_correct"] = batch_full_labeled_correct

                unlabeled_correct += metric["unlabeled_correct"]
                full_unlabeled_correct += metric["full_unlabeled_correct"]
                labeled_correct += metric["labeled_correct"]
                full_labeled_correct += metric["full_labeled_correct"]
                rel_correct += num_rel_correct
                rel_confusion_matrix += rel_conf_mat

                total_sentences += metric["total_sentences"]
                total_words += metric["total_words"]

                pos_loss += loss["pos_loss"]
                arc_loss += loss["arc_loss"]
                rel_loss += rel_loss_dict["rel_loss"]

                seen_batch += 1

            if total_words > 0.0:
                unlabeled_attachment_score = unlabeled_correct / total_words
                labeled_attachment_score = labeled_correct / total_words
                rel_accuracy = rel_correct / total_words

            if total_sentences > 0.0:
                unlabeled_full_match = full_unlabeled_correct / total_sentences
                labeled_full_match = full_labeled_correct / total_sentences

        accuracy_score = {
            "uas": unlabeled_attachment_score,
            "las": labeled_attachment_score,
            "ufm": unlabeled_full_match,
            "lfm": labeled_full_match,
            "rel_acc": rel_accuracy,
            "rel_confusion_matrix": rel_confusion_matrix.tolist()
        }

        d_tag = "S_" if domain == "source" else "T_"

        accuracy_log_header = "{:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(
            f"{d_tag}UAS",
            f"{d_tag}LAS",
            f"{d_tag}UFM",
            f"{d_tag}LFM",
            f"{d_tag}REL_ACC",
        )
        accuracy_log_line = "{:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(
            accuracy_score["uas"],
            accuracy_score["las"],
            accuracy_score["ufm"],
            accuracy_score["lfm"],
            accuracy_score["rel_acc"]
        )

        loss_score = {
            "pos_loss": pos_loss / seen_batch,
            "arc_loss": arc_loss / seen_batch,
            "rel_loss": rel_loss / seen_batch,
        }
        total_loss = loss_score["pos_loss"] + loss_score["arc_loss"] + loss_score["rel_loss"]
        loss_log_header = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            f"{d_tag}TOTAL_LOSS", f"{d_tag}POS_LOSS", f"{d_tag}ARC_LOSS", f"{d_tag}REL_LOSS",
        )
        loss_log_line = "{:<20} | {:<20} | {:<20} | {:<20}".format(
            total_loss, loss_score["pos_loss"], loss_score["arc_loss"], loss_score["rel_loss"]
        )

        accuracy_result = Result(
            metric_score=rel_accuracy,
            log_header=accuracy_log_header,
            log_line=accuracy_log_line,
            metric_detail=accuracy_score
        )

        loss_result = Result(
            metric_score=total_loss,
            log_header=loss_log_header,
            log_line=loss_log_line,
            metric_detail=loss_score
        )

        return accuracy_result, loss_result

    def evaluate_probe(
            self,
            dataloader: DataLoader,
            protos_dict: Dict,
            domain: str = "target",
    ):
        unlabeled_correct = 0.0
        labeled_correct = 0.0
        num_rel = protos_dict["ys_proto"].size(0)
        rel_correct = 0.0
        rel_confusion_matrix = np.zeros((num_rel, num_rel))

        total_words = 0.0
        total_sentences = 0.0

        arc_loss = 0.0
        rel_loss = 0.0
        rel_accuracy = 0.0

        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        # seen_batch = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                # embedding batch
                batch = self.encoder.embed(batch)
                metric, loss, correct_heads, mask = self.arc_biaffine.evaluate(batch=batch)  # noqa
                batch_rel = self.embed_rel(batch=batch)

                metric_rel, loss_rel = self.probe_module.evaluate(batch=batch_rel, protos=protos_dict)
                correct_labels, mask = metric_rel["correct_labels"], metric_rel["mask"]

                correct_labels_and_indices = correct_heads * correct_labels
                batch_labeled_correct = correct_labels_and_indices.sum().item()

                metric["labeled_correct"] = batch_labeled_correct

                unlabeled_correct += metric["unlabeled_correct"]
                labeled_correct += metric["labeled_correct"]

                rel_correct += metric_rel["num_correct"]
                rel_confusion_matrix += metric_rel["conf_mat"]

                total_sentences += metric["total_sentences"]
                total_words += metric["total_words"]

                arc_loss += loss["arc_loss"] * metric["total_words"]
                rel_loss += loss_rel["loss_clf"] * metric["total_words"]

            if total_words > 0.0:
                unlabeled_attachment_score = unlabeled_correct / total_words
                labeled_attachment_score = labeled_correct / total_words
                rel_accuracy = rel_correct / total_words

        accuracy_score = {
            "uas": unlabeled_attachment_score,
            "las": labeled_attachment_score,
            "rel_acc": rel_accuracy,
            "rel_confusion_matrix": rel_confusion_matrix.tolist()
        }

        d_tag = "S_" if domain == "source" else "T_"
        accuracy_log_header = "{:<20} | {:<20} | {:<20}".format(
            f"{d_tag}UAS",
            f"{d_tag}LAS",
            f"{d_tag}REL_ACC",
        )
        accuracy_log_line = "{:<20} | {:<20} | {:<20}".format(
            accuracy_score["uas"],
            accuracy_score["las"],
            accuracy_score["rel_acc"]
        )

        loss_score = {
            "arc_loss": arc_loss / total_words,
            "rel_loss": rel_loss / total_words
        }
        total_loss = loss_score["arc_loss"] + loss_score["rel_loss"]
        loss_log_header = "{:<20} | {:<20} | {:<20}".format(
            f"{d_tag}TOTAL_LOSS", f"{d_tag}ARC_LOSS", f"{d_tag}REL_LOSS",
        )
        loss_log_line = "{:<20} | {:<20} | {:<20}".format(
            total_loss, loss_score["arc_loss"], loss_score["rel_loss"]
        )
        accuracy_result = Result(
            metric_score=rel_accuracy,
            log_header=accuracy_log_header,
            log_line=accuracy_log_line,
            metric_detail=accuracy_score
        )
        loss_result = Result(
            metric_score=rel_loss,
            log_header=loss_log_header,
            log_line=loss_log_line,
            metric_detail=loss_score
        )

        return accuracy_result, loss_result

    def predict(self, dataloader: DataLoader, save_dir):
        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                output_list = self.encoder.predict(batch, save_dir=save_dir)

    def get_s_protos(self, dataloader: DataLoader):
        s_protos = OrderedDict()
        Xs, Ys = [], []
        with torch.no_grad():
            self.encoder.eval()
            for batch in tqdm(dataloader, leave=False, desc="[estimating prototypes]"):
                batch = self.encoder.embed(batch)
                batch = self.embed_rel(batch=batch)
                x = batch["rel_representations"]
                y = batch["deprel_ids"]
                mask = y != elephant.config.pad_label_id
                x = x[mask].detach().cpu()
                y = y[mask].detach().cpu()
                Xs.append(x)
                Ys.append(y)
            Xs = torch.cat(Xs)
            Ys = torch.cat(Ys)
            unique_classes = torch.unique(Ys)
            for c in unique_classes:
                class_proto = Xs[Ys == c].mean(dim=0)  # torch.index_select(Xs, 0, self._extract_class_indices(Ys, c))
                s_protos[c.item()] = class_proto.detach()
        s_protos_dict = {
            "xs_proto": torch.stack(list(s_protos.values())).squeeze(1).to(elephant.device),
            "ys_proto": torch.LongTensor(list(s_protos.keys())).to(elephant.device)
        }
        return s_protos_dict

    @staticmethod
    def embed_rel(batch: Dict):
        x = batch["embedding"]
        head_ids = batch["head_ids"]

        batch_size = x.size()[0]
        range_vector = torch.arange(batch_size, device=elephant.device).unsqueeze(1)

        selected_head_tag = x[range_vector, head_ids]
        selected_head_tag = selected_head_tag.contiguous()

        rel_representations = selected_head_tag - x
        batch["rel_representations"] = rel_representations
        return batch
