from .template import DataProducerTemplate as DataProducer
from .ud_mbert_dp import UDMBertDPDataProducer
from .hdf5_producer import HDF5DataProducer


def get_data_producer(producer_cfg) -> DataProducer:
    producer = {
        "ud_mbert_dp": UDMBertDPDataProducer,
        "hdf5_producer": HDF5DataProducer
    }[producer_cfg.name](producer_cfg)

    return producer


__all__ = [
    "DataProducer",
    "UDMBertDPDataProducer",
    "get_data_producer",
    "HDF5DataProducer",
]
