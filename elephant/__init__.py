import os
from pathlib import Path

import easydict
import torch
from transformers import set_seed

from elephant.utils.dist_utils import init_dist, gpu_to_use
from elephant.utils.config_utils import config_from_yaml_file

# set global random seed
set_seed(0)

# disable parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# global variable: version
__version__ = "0.1"

# global variable: config
config_file = "./config/main.yaml"  # default: "./config/main.yaml"; plot: "../config/plot.yaml"
config = config_from_yaml_file(config_file, easydict.EasyDict())

# global variable: root_path
root_path = (Path(__file__) / "../../").resolve()

data_root = root_path / config.data_root
data_root.mkdir(exist_ok=True, parents=True)

corpus_path = data_root / config.corpus_path
corpus_path.mkdir(exist_ok=True, parents=True)

cache_path = data_root / config.cache_path
corpus_path.mkdir(exist_ok=True, parents=True)

model_root = root_path / config.model_root
model_root.mkdir(exist_ok=True, parents=True)

tokenizer_root = root_path / config.tokenizer_root
tokenizer_root.mkdir(exist_ok=True, parents=True)

output_root = root_path / config.output_root
output_root.mkdir(exist_ok=True, parents=True)

checkpoint_path = output_root / config.checkpoint_path
checkpoint_path.mkdir(parents=True, exist_ok=True)

log_path = output_root / config.log_path
log_path.mkdir(parents=True, exist_ok=True)

tensorboard_path = output_root / config.tensorboard_path
tensorboard_path.mkdir(parents=True, exist_ok=True)

# global variable: device
if not torch.cuda.is_available():
    device = torch.device("cpu")
elif not config.trainer.dist:
    gpus = gpu_to_use()
    if len(gpus) == 0:
        device = torch.device("cpu")
    else:
        device = gpus[-1]  # gpus[-1]  torch.device("cuda:5")
else:
    init_dist(launcher=config.trainer.launcher, backend="nccl")
    device = torch.cuda.current_device()

__all__ = [
    "__version__",
    "root_path",
    "data_root",
    "corpus_path",
    "cache_path",
    "model_root",
    "tokenizer_root",
    "output_root",
    "checkpoint_path",
    "log_path",
    "tensorboard_path",
    "device",
    "config",
]
