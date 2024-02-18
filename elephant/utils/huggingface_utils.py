from pathlib import Path
from typing import Dict

from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import adapters
from transformers import set_seed as hf_set_seed


def set_seed(seed: int):
    hf_set_seed(seed)


def get_huggingface_model(
        model_name: str,
        model_path: Path,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
):
    if not model_path.exists() or not any(model_path.iterdir()):
        config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        model = AutoModel.from_pretrained(model_name, config=config)
        config.save_pretrained(model_path)
        model.save_pretrained(model_path)
    else:
        config = AutoConfig.from_pretrained(
            model_path, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        model = AutoModel.from_pretrained(model_path, config=config)
    return model


def get_huggingface_tokenizer(tokenizer_name: str, tokenizer_path: Path):
    if not tokenizer_path.exists() or not any(tokenizer_path.iterdir()):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def get_adapter_config(adapter_cfg: Dict):
    try:
        adapter_config = {
            "HoulsbyConfig": adapters.HoulsbyConfig,
            "HoulsbyInvConfig": adapters.HoulsbyInvConfig,
            "PfeifferConfig": adapters.PfeifferConfig,
            "PfeifferInvConfig": adapters.PfeifferInvConfig,
        }[adapter_cfg["config_name"]]()
    except KeyError:  #
        adapter_config = adapters.AdapterConfig(
            mh_adapter=False,
            output_adapter=True,
            reduction_factor=adapter_cfg["reduction_factor"],
            non_linearity="relu",
            original_ln_before=True,
        )

    return adapter_config
