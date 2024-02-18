from typing import Dict

import torch


class EncoderTemplate(torch.nn.Module):
    """
    Abstract base class for encoding data. Design to cooperate with pytorch dataset, will be called by pytorch dataset
    to preprocess dataset for specific encoder.
    """

    def __init__(self, encoder_cfg):
        super(EncoderTemplate, self).__init__()

        self.encoder_cfg = encoder_cfg

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    def embed(self, batch: Dict) -> Dict:
        """
        Add word embedding
        """
        raise NotImplementedError
