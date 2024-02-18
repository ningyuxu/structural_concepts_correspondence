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


class MBertProbePOSModel(ModelTemplate):
    def __init__(self, model_cfg):
        super(MBertProbePOSModel, self).__init__(model_cfg)
        self.encoder = get_encoder(self.model_cfg.encoder)
        self.pos_predictor = get_task(self.model_cfg.task.pos_module)
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
        loss, count, detail_loss = self.pos_predictor.forward_loss(batch=batch)

        log_header = "{:<20}".format("pos_loss")
        log_line = "{:<20}".format(loss.item())
        loss_result = Result(
            metric_score=loss,
            log_header=log_header,
            log_line=log_line,
            metric_detail=detail_loss,
        )
        return loss_result, count

    def forward_loss_probe(self, batch: Dict, protos_dict: Dict):
        batch = self.encoder.embed(batch)
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
            # is_ft: bool = False,
            adapt: bool = False,
            adapter_func: Callable = None
    ) -> Tuple[Result, Result]:
        pos_correct = 0.0
        pos_confusion_matrix = np.zeros((self.pos_predictor.num_pos, self.pos_predictor.num_pos))
        # total_sentences = 0.0
        total_words = 0.0

        pos_loss = 0.0
        pos_accuracy = 0.0

        with torch.no_grad():
            seen_batch = 0
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)
                correct_labels, num_pos_correct, loss_dict, mask, pos_conf_mat = self.pos_predictor.evaluate(
                    batch=batch, adapt=adapt, adapter_func=adapter_func
                )

                pos_correct += num_pos_correct
                pos_confusion_matrix += pos_conf_mat

                total_words += correct_labels.numel() - (1 - mask.long()).sum().item()

                pos_loss += loss_dict["pos_loss"]

                seen_batch += 1

            if total_words > 0.0:
                pos_accuracy = pos_correct / total_words

        accuracy_score = {
            "pos_acc": pos_accuracy,
            "pos_confusion_matrix": pos_confusion_matrix.tolist()
        }

        d_tag = "S_" if domain == "source" else "T_"

        accuracy_log_header = "{:<20}".format(f"{d_tag}POS_ACC")
        accuracy_log_line = "{:<20}".format(accuracy_score["pos_acc"])

        loss_score = {"pos_loss": pos_loss / seen_batch}
        total_loss = loss_score["pos_loss"]
        loss_log_header = "{:<20}".format(f"{d_tag}POS_LOSS")
        loss_log_line = "{:<20}".format(total_loss)

        accuracy_result = Result(
            metric_score=pos_accuracy,
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
        num_pos = protos_dict["ys_proto"].size(0)
        pos_correct = 0.0
        pos_confusion_matrix = np.zeros((num_pos, num_pos))

        total_words = 0.0

        pos_loss = 0.0
        pos_accuracy = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                # embedding batch
                batch = self.encoder.embed(batch)
                metric, loss = self.probe_module.evaluate(batch=batch, protos=protos_dict)
                correct_labels, mask = metric["correct_labels"], metric["mask"]
                words = correct_labels.numel() - (1 - mask.long()).sum().item() + metric["num_ignored_valid_words"]
                total_words += words
                pos_correct += metric["num_correct"]
                pos_confusion_matrix += metric["conf_mat"]
                pos_loss += loss["loss_clf"] * words

            if total_words > 0.0:
                pos_accuracy = pos_correct / total_words

        accuracy_score = {
            "pos_acc": pos_accuracy,
            "pos_confusion_matrix": pos_confusion_matrix.tolist()
        }

        d_tag = "S_" if domain == "source" else "T_"

        accuracy_log_header = "{:<20}".format(f"{d_tag}POS_ACC")
        accuracy_log_line = "{:<20}".format(accuracy_score["pos_acc"])

        loss_score = {"pos_loss": pos_loss / total_words}
        total_loss = loss_score["pos_loss"]
        loss_log_header = "{:<20}".format(f"{d_tag}POS_LOSS")
        loss_log_line = "{:<20}".format(total_loss)

        accuracy_result = Result(
            metric_score=pos_accuracy,
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

    def predict(self, dataloader: DataLoader, save_dir):
        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                output_list = self.encoder.predict(batch, save_dir=save_dir)

    def get_s_protos(self, dataloader: DataLoader):
        s_protos = OrderedDict()
        Xs, Ys = [], []
        with torch.no_grad():
            self.encoder.eval()
            for batch in tqdm(dataloader, leave=False):
                batch = self.encoder.embed(batch)
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
                s_protos[c.item()] = class_proto.detach()
        s_protos_dict = {
            "xs_proto": torch.stack(list(s_protos.values())).squeeze(1).to(elephant.device),
            "ys_proto": torch.LongTensor(list(s_protos.keys())).to(elephant.device)
        }
        return s_protos_dict
