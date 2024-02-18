from typing import Dict, Tuple, Optional

import torch


class TaskModelTemplate(torch.nn.Module):
    def __init__(self, task_cfg):
        super(TaskModelTemplate, self).__init__()

        self.task_cfg = task_cfg

    def forward_loss(self, **kwargs) -> Tuple[torch.Tensor, int, Optional[Dict]]:
        raise NotImplementedError

    def evaluate(self, **kwargs) -> Tuple[Dict, Dict]:
        raise NotImplementedError
