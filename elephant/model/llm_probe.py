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


class LLMProbeModel(ModelTemplate):
    def __init__(self, model_cfg):
        super(LLMProbeModel, self).__init__(model_cfg)
        self.probe_module = get_task(self.model_cfg.task.probe_module)
        self.task = self.model_cfg.task.probe_module.task

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
        batch = self.probe_module.embed(batch)
        return batch

    def forward_loss(self, batch: Dict, protos_dict: Dict) -> Tuple[Result, int]:
        # if self.task == "rel":
        #     batch = self.embed_rel(batch)
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
            protos_dict: Dict,
            domain: str = "target",
    ):
        num_cls = torch.unique(protos_dict["ys_proto"]).size(0)
        cls_correct = 0.0
        cls_confusion_matrix = np.zeros((num_cls, num_cls))
        total_words = 0.0

        cls_loss = 0.0
        cls_accuracy = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False):
                # if self.task == "rel":
                #     batch = self.embed_rel(batch)
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
            f"{self.task}_confusion_matrix": cls_confusion_matrix.tolist(),
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

    def get_s_protos(self, dataloader: DataLoader):
        s_protos = OrderedDict()
        Xs, Ys = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, leave=False, desc="[estimating prototypes]"):
                if self.task == "rel":
                    # batch = self.embed_rel(batch)
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
