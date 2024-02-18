from tqdm import tqdm
from typing import Tuple, Dict, Callable, List
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import elephant
from elephant.data import DataLoader
from elephant.trainer.result import Result
from elephant.utils.logging_utils import get_logger

from elephant.model.encoder import get_encoder
from .header import get_task
from .template import ModelTemplate


logger = get_logger("elephant")


class LSAdaProtoModel(ModelTemplate):
    def __init__(self, model_cfg):
        super(LSAdaProtoModel, self).__init__(model_cfg)
        self.encoder = get_encoder(self.model_cfg.encoder)
        self.parser = get_task(self.model_cfg.task.parser_module)
        self.probe_module = get_task(self.model_cfg.task.probe_module)
        self.proto_module = get_task(self.model_cfg.task.proto_module)
        self.task = self.model_cfg.task.probe_module.task

    def activate_adapter(self, adapter_name: str):
        self.encoder.activate_adapter(adapter_name)

    def freeze_adapter(self, adapter_name: str):
        self.encoder.freeze_adapter(adapter_name)

    def reset_adaptor_params(self):
        self.proto_module.reset_adaptor_params()

    def freeze_proto_module(self, freeze: bool = True):
        self.proto_module.freeze_module(freeze)

    def freeze_probe_module(self, freeze: bool = True):
        self.probe_module.freeze_module(freeze)

    def save_adaptor(self, save_path: Path):
        self.proto_module.save_adaptor(save_path=save_path)

    def load_adaptor(self, save_path: Path):
        self.proto_module.load_adaptor(save_path=save_path)

    def orthogonalize_adaptor(self, meta_train: bool = False, idx: int = None):
        self.proto_module.orthogonalize_adaptor(meta_train=meta_train, idx=idx)

    def embed_data(self, batch: Dict) -> Dict:
        """
        Call encoder to embed data
        """
        batch = self.encoder.embed(batch)
        return batch

    def forward_loss_parser(self, batch) -> Tuple[Result, int]:
        batch = self.encoder.embed(batch)
        loss, count, detail_loss = self.parser.forward_loss(batch=batch)
        log_header = "{:<20} | {:<20} | {:<20}".format(
            "total_loss", "arc_loss", "rel_loss"
        )
        log_line = "{:<20} | {:<20} | {:<20}".format(
            loss.item(),
            detail_loss["arc_loss"],
            detail_loss["rel_loss"]
        )
        loss_result = Result(
            metric_score=loss,
            log_header=log_header,
            log_line=log_line,
            metric_detail=detail_loss,
        )
        return loss_result, count

    def forward_loss(
            self,
            support_dataloader: DataLoader,
            query_dataloader: DataLoader,
            s_protos_dict: Dict,
            idx: int,
            valid_labels: List,
    ) -> Tuple[Result, int]:
        samples = self._get_sq_from_dataloader(support_dataloader, query_dataloader)
        valid_protos_dict = self.get_valid_protos_dict(s_protos_dict=s_protos_dict, valid_labels=valid_labels)
        samples.update(valid_protos_dict)

        loss, count, detail_loss = self.proto_module.forward_loss(samples=samples, idx=idx)

        log_header = "{:<20} | {:<20}".format("loss_val", "acc_val")
        log_line = "{:<20} | {:<20}".format(detail_loss["loss_val"], detail_loss["acc_val"])
        loss_result = Result(
            metric_score=loss,
            log_header=log_header,
            log_line=log_line,
            metric_detail=detail_loss,
        )
        return loss_result, count

    def forward_loss_adaptor(self, batch: Dict, s_protos_dict: Dict):
        batch = self.encoder.embed(batch)
        if self.task == "rel":
            batch = self.embed_rel(batch)
        loss, count, detail_loss = self.proto_module.forward_loss_adaptor(batch=batch, protos=s_protos_dict)

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

    def forward_loss_probe(self, batch: Dict, protos_dict: Dict):
        batch = self.encoder.embed(batch)
        if self.task == "rel":
            batch = self.embed_rel(batch)
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
            support_set: Dict,
            s_protos_dict: Dict,
            valid_labels: List,
            meta_train: bool = False,
            idx: int = None,
    ) -> Tuple[Result, Result]:
        valid_protos_dict = self.get_valid_protos_dict(s_protos_dict=s_protos_dict, valid_labels=valid_labels)
        # num_cls = torch.unique(valid_protos_dict["ys_proto"]).size(0)
        cls_correct = 0.0
        # cls_confusion_matrix = np.zeros((num_cls, num_cls))
        total_words = 0.0

        cls_loss = 0.0
        cls_accuracy = 0.0
        init_confmat = True

        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)
                if self.task == "rel":
                    batch = self.embed_rel(batch)
                batch.update(support_set)
                batch.update(valid_protos_dict)
                metric, loss = self.proto_module.evaluate(samples=batch, meta_train=meta_train, idx=idx)
                correct_labels, mask = metric["correct_labels"], metric["mask"]

                words = correct_labels.numel() - (1 - mask.long()).sum().item() + metric["num_ignored_valid_words"]
                total_words += words

                cls_correct += metric["num_correct"]

                if init_confmat:
                    cls_confusion_matrix = np.zeros_like(metric["conf_mat"])
                    init_confmat = False

                cls_confusion_matrix += metric["conf_mat"]
                cls_loss += loss["loss_clf"] * words

            if total_words > 0.0:
                cls_accuracy = cls_correct / total_words

        accuracy_score = {
            f"{self.task}_acc": cls_accuracy,
            f"{self.task}_confusion_matrix": cls_confusion_matrix.tolist()
        }

        accuracy_log_header = "{:<20}".format(f"{self.task.upper()}_ACC")
        accuracy_log_line = "{:<20}".format(accuracy_score[f"{self.task}_acc"])

        loss_score = {f"{self.task}_loss": cls_loss / total_words}
        total_loss = loss_score[f"{self.task}_loss"]
        loss_log_header = "{:<20}".format(f"{self.task.upper()}_LOSS")
        loss_log_line = "{:<20}".format(total_loss)

        accuracy_result = Result(
            metric_score=cls_accuracy,
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

    def evaluate_parser(self, dataloader: DataLoader) -> Tuple[Result, Result]:
        unlabeled_correct = 0.0
        full_unlabeled_correct = 0.0
        labeled_correct = 0.0
        full_labeled_correct = 0.0

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
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)
                metric, loss = self.parser.evaluate(batch=batch)

                unlabeled_correct += metric["unlabeled_correct"]
                full_unlabeled_correct += metric["full_unlabeled_correct"]
                labeled_correct += metric["labeled_correct"]
                full_labeled_correct += metric["full_labeled_correct"]

                total_sentences += metric["total_sentences"]
                total_words += metric["total_words"]

                pos_loss += loss["pos_loss"] * metric["total_words"]
                arc_loss += loss["arc_loss"] * metric["total_words"]
                rel_loss += loss["rel_loss"] * metric["total_words"]

            if total_words > 0.0:
                unlabeled_attachment_score = unlabeled_correct / total_words
                labeled_attachment_score = labeled_correct / total_words
            if total_sentences > 0.0:
                unlabeled_full_match = full_unlabeled_correct / total_sentences
                labeled_full_match = full_labeled_correct / total_sentences

        accuracy_score = {
            "uas": unlabeled_attachment_score,
            "las": labeled_attachment_score,
            "ufm": unlabeled_full_match,
            "lfm": labeled_full_match,
        }

        accuracy_log_header = "{:<18} | {:<18} | {:<18} | {:<18}".format(
            f"UAS",
            f"LAS",
            f"UFM",
            f"LFM",
        )
        accuracy_log_line = "{:<18} | {:<18} | {:<18} | {:<18}".format(
            accuracy_score["uas"],
            accuracy_score["las"],
            accuracy_score["ufm"],
            accuracy_score["lfm"],
        )

        loss_score = {
            "pos_loss": pos_loss / total_words,
            "arc_loss": arc_loss / total_words,
            "rel_loss": rel_loss / total_words,
        }
        total_loss = loss_score["pos_loss"] + loss_score["arc_loss"] + loss_score["rel_loss"]
        loss_log_header = "{:<18} | {:<18} | {:<18} | {:<18}".format(
            f"TOTAL_LOSS", f"POS_LOSS", f"ARC_LOSS", f"REL_LOSS",
        )
        loss_log_line = "{:<18} | {:<18} | {:<18} | {:<18}".format(
            total_loss, loss_score["pos_loss"], loss_score["arc_loss"], loss_score["rel_loss"]
        )

        # if self.model_cfg.metric_score == "uas":
        #     metric_score = unlabeled_attachment_score
        # else:
        metric_score = labeled_attachment_score

        accuracy_result = Result(
            metric_score=metric_score,
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
            domain: str = "source",
    ):
        num_cls = protos_dict["ys_proto"].size(0)
        cls_correct = 0.0
        cls_confusion_matrix = np.zeros((num_cls, num_cls))

        total_words = 0.0

        cls_loss = 0.0
        cls_accuracy = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                # embedding batch
                batch = self.encoder.embed(batch)
                if self.task == "rel":
                    batch = self.embed_rel(batch)
                metric, loss = self.probe_module.evaluate(batch=batch, protos=protos_dict)
                correct_labels, mask = metric["correct_labels"], metric["mask"]
                words = correct_labels.numel() - (1 - mask.long()).sum().item() + metric["num_ignored_valid_words"]
                total_words += words
                cls_correct += metric["num_correct"]
                cls_confusion_matrix += metric["conf_mat"]
                cls_loss += loss["loss_clf"] * words

            if total_words > 0.0:
                cls_accuracy = cls_correct / total_words

        accuracy_score = {
            f"{self.task}_acc": cls_accuracy,
            f"{self.task}_confusion_matrix": cls_confusion_matrix.tolist()
        }

        d_tag = "S_" if domain == "source" else "T_"

        accuracy_log_header = "{:<20}".format(f"{d_tag}{self.task.upper()}_ACC")
        accuracy_log_line = "{:<20}".format(accuracy_score[f"{self.task}_acc"])

        loss_score = {f"{self.task}_loss": cls_loss / total_words}
        total_loss = loss_score[f"{self.task}_loss"]
        loss_log_header = "{:<20}".format(f"{d_tag}{self.task.upper()}_LOSS")
        loss_log_line = "{:<20}".format(total_loss)

        accuracy_result = Result(
            metric_score=cls_accuracy,
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

    def get_support_set_for_eval(self, dataloader: DataLoader):
        Xs, Ys = self._get_embedding_from_dataloader(dataloader=dataloader)
        support_dict = {
            "x_support": Xs.to(elephant.device),
            "y_support": Ys.to(elephant.device)
        }
        return support_dict

    def predict(self, dataloader: DataLoader, save_dir):
        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                output_list = self.encoder.predict(batch, save_dir=save_dir)

    def get_s_protos(self, dataloader: DataLoader, proj: bool = False, requires_grad: bool = False):
        s_protos = OrderedDict()
        Xs, Ys = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)
                if self.task == "rel":
                    batch = self.embed_rel(batch)
                    x = batch["rel_representations"]
                    y = batch["deprel_ids"]
                else:
                    x = batch["embedding"]
                    y = batch["postag_ids"]
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
                if proj:
                    class_proto = self.probe_module.proj(class_proto.to(elephant.device))
                if requires_grad:
                    s_protos[c.item()] = class_proto
                else:
                    s_protos[c.item()] = class_proto.detach()

        s_protos_dict = {
            "xs_proto": torch.stack(list(s_protos.values())).squeeze(1).to(elephant.device),
            "ys_proto": torch.LongTensor(list(s_protos.keys())).to(elephant.device)
        }
        return s_protos_dict

    def _get_embedding_from_dataloader(self, dataloader: DataLoader):
        Xs = []
        Ys = []
        for batch in dataloader:
            batch = self.encoder.embed(batch)
            if self.task == "rel":
                batch = self.embed_rel(batch)
                xs = batch["rel_representations"]
                ys = batch["deprel_ids"]
            else:
                xs = batch["embedding"]
                ys = batch["postag_ids"]
            mask = ys != elephant.config.pad_label_id
            xs = xs[mask].detach()
            ys = ys[mask].detach()
            Xs.append(xs)
            Ys.append(ys)
        Xs = torch.cat(Xs, dim=0)
        Ys = torch.cat(Ys, dim=0)
        return Xs, Ys

    def _get_sq_from_dataloader(self, support_dataloader: DataLoader, query_dataloader: DataLoader):
        samples = dict()
        Xs, Ys = self._get_embedding_from_dataloader(support_dataloader)
        samples["x_support"] = Xs
        samples["y_support"] = Ys
        Xq, Yq = self._get_embedding_from_dataloader(query_dataloader)
        samples["x_query"] = Xq
        samples["y_query"] = Yq
        return samples

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

    @staticmethod
    def get_valid_protos_dict(s_protos_dict: Dict, valid_labels: List):
        ys = s_protos_dict["ys_proto"].detach().cpu().tolist()
        valid_labels = set(valid_labels)
        valid_idx = [i for i, y in enumerate(ys) if y in valid_labels]
        x_protos = s_protos_dict["xs_proto"].detach().cpu()[valid_idx]
        y_protos = torch.LongTensor(ys)[valid_idx]
        valid_protos_dict = {
            "xs_proto": x_protos.to(elephant.device),
            "ys_proto": y_protos.to(elephant.device)
        }
        return valid_protos_dict
