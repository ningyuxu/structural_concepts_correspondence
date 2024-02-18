from .template import EncoderTemplate as Encoder
from .mbert import MBertLATAEncoder


def get_encoder(encoder_cfg) -> Encoder:
    encoder = {
        "mbert_with_lata": MBertLATAEncoder
    }[encoder_cfg.name](encoder_cfg)
    return encoder


__all__ = [
    "get_encoder",
    "Encoder",
    "MBertLATAEncoder",
]
