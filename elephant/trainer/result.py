from typing import Dict, Union

import torch


class Result(object):
    def __init__(
            self,
            metric_score: Union[float, torch.Tensor],
            log_header: str,
            log_line: str,
            metric_detail: Dict
    ):
        self.metric_score: Union[float, torch.Tensor] = metric_score
        self.log_header: str = log_header
        self.log_line: str = log_line
        self.metric_detail = metric_detail
