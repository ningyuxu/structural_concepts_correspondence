from typing import Dict

import torch

import elephant
from elephant.utils.huggingface_utils import (
    get_huggingface_tokenizer, get_huggingface_model, get_adapter_config
)
from elephant.utils.torch_utils import freeze_module
from elephant.utils.predict_utils import save_as_hdf5

from .template import EncoderTemplate


class MBertLATAEncoder(EncoderTemplate):
    def __init__(self, encoder_cfg):
        super(MBertLATAEncoder, self).__init__(encoder_cfg)

        self.tokenizer = self.get_tokenizer()
        self.pretrained_model = self.get_pretrained_model()
        self.freeze_pretrained_model(self.encoder_cfg.pretrained_model.freeze_layer)

        # self.add_adapter("lang_adapter", self.encoder_cfg.lang_adapter)
        # self.add_adapter("task_adapter", self.encoder_cfg.task_adapter)

    @property
    def embedding_dim(self) -> int:
        return self.pretrained_model.config.hidden_size

    def get_tokenizer(self):
        tokenizer_name = self.encoder_cfg.tokenizer.name
        assert tokenizer_name.startswith("huggingface")

        hf_tokenizer_name = tokenizer_name.split('/', 2)[1]
        hf_tokenizer_path = elephant.root_path / "tokenizer" / tokenizer_name
        tokenizer = get_huggingface_tokenizer(hf_tokenizer_name, hf_tokenizer_path)
        return tokenizer

    def get_pretrained_model(self):
        model_name = self.encoder_cfg.pretrained_model.name
        assert model_name.startswith("huggingface")

        hf_model_name = model_name.split('/', 2)[1]
        hf_model_path = elephant.root_path / "model" / model_name
        model = get_huggingface_model(hf_model_name, hf_model_path)
        # model.apply(model._init_weights)

        return model

    def freeze_pretrained_model(self, freeze_layer: int = -1) -> None:
        if freeze_layer == -1:
            return  # do not freeze
        if freeze_layer >= 0:
            for i in range(freeze_layer + 1):
                if i == 0:
                    self._freeze_embedding_layer()
                else:
                    self._freeze_encoder_layer(i)

    def activate_pretrained_model(self) -> None:
        for i in range(self.num_hidden_layers + 1):
            if i == 0:
                self._freeze_embedding_layer(freeze=False)
            else:
                self._freeze_encoder_layer(layer=i, freeze=False)

    def _freeze_embedding_layer(self, freeze: bool = True) -> None:
        freeze_module(self.pretrained_model.embeddings, freeze=freeze)

    def _freeze_encoder_layer(self, layer: int, freeze: bool = True) -> None:
        freeze_module(self.pretrained_model.encoder.layer[layer - 1], freeze=freeze)

    def add_adapter(self, name: str, adapter_cfg: Dict) -> None:
        config = get_adapter_config(adapter_cfg)
        self.pretrained_model.add_adapter(name, config)

    def activate_adapter(self, adapter_name: str) -> None:
        self.pretrained_model.set_active_adapters(adapter_name)

    def freeze_adapter(self, adapter_name: str) -> None:
        for m in self.pretrained_model.encoder.layer:
            if hasattr(m.attention.output.adapters, adapter_name):
                freeze_module(getattr(m.attention.output.adapters, adapter_name))
            if hasattr(m.output.adapters, adapter_name):
                freeze_module(getattr(m.output.adapters, adapter_name))

    # --------------------------------------------------------------------------------------------
    # encode data
    # --------------------------------------------------------------------------------------------
    def embed(self, batch: Dict) -> Dict:
        """
        Add word embedding
        """
        lang = batch["lang"]
        genre = batch["genre"]
        split = batch["split"]
        seqid = batch["seqid"]
        tokens = batch["tokens"]
        length = batch["length"]
        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        postag_ids = batch["postag_ids"]
        head_ids = batch["head_ids"]
        deprel_ids = batch["deprel_ids"]
        # distances = batch["distances"]
        # deprel_matrix = batch["deprel_matrix"]

        output = self.pretrained_model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_states = torch.stack(output["hidden_states"])
        embedding = hidden_states[self.encoder_cfg.embedding.layer]

        batch_with_embedding = {
            "lang": lang,
            "genre": genre,
            "split": split,
            "seqid": seqid,
            "tokens": tokens,
            "length": length,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "postag_ids": postag_ids,
            "head_ids": head_ids,
            "deprel_ids": deprel_ids,
            # "distances": distances,
            # "deprel_matrix": deprel_matrix,
            "embedding": embedding,
        }

        return batch_with_embedding

    def predict(self, batch: Dict, save_dir):
        lang = batch["lang"]
        genre = batch["genre"]
        split = batch["split"]
        seqid = batch["seqid"]
        tokens = batch["tokens"]
        length = batch["length"]
        postag_ids = batch["postag_ids"]
        head_ids = batch["head_ids"]
        deprel_ids = batch["deprel_ids"]

        token_ids = batch["token_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]

        output = self.pretrained_model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        hidden_states = torch.stack(output["hidden_states"], dim=0)
        hidden_states = torch.permute(hidden_states, (1, 0, 2, 3))
        output_list = []

        for idx in range(len(lang)):
            length_s = length[idx]
            seqid_s = seqid[idx]
            lang_s = lang[idx]
            genre_s = genre[idx]
            split_s = split[idx]
            tokens_s = tokens[idx][:length_s]
            token_ids_s = token_ids[idx][:length_s]
            postag_ids_s = postag_ids[idx][:length_s]
            head_ids_s = head_ids[idx][:length_s]
            arc_label_ids_s = deprel_ids[idx][:length_s]
            hidden_states_s = hidden_states[idx][:, :length_s]

            output = {
                "seqid": seqid_s,
                "lang": lang_s,
                "genre": genre_s,
                "split": split_s,
                "tokens": tokens_s,
                "token_ids": token_ids_s,
                "length": length_s,
                "pos_label_ids": postag_ids_s,
                "head_ids": head_ids_s,
                "arc_label_ids": arc_label_ids_s,
                "hidden_state": hidden_states_s
            }
            save_as_hdf5(output, save_dir=save_dir)
            output_list.append(output)
        return output_list
