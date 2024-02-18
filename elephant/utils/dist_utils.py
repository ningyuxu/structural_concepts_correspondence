"""
Modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/dist_utils.py
"""

import os
import subprocess
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .logging_utils import get_logger

logger = get_logger("elephant")


def init_dist(launcher: str, backend: str = "nccl") -> None:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    if launcher == "pytorch":
        _init_dist_pytorch(backend)
    else:
        raise ValueError(f"Invalid launcher type: {launcher}")


def gpu_to_use() -> List[int]:
    gpus = _get_free_gpus(max_gpus=torch.cuda.device_count())
    gpus_to_use = [int(gpu) for gpu in gpus]
    return gpus_to_use


def _init_dist_pytorch(backend: str) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        gpus = _get_free_gpus(max_gpus=torch.cuda.device_count())
    gpus_to_use = [int(gpu) for gpu in gpus]

    torch.cuda.set_device(gpus_to_use[local_rank % len(gpus_to_use)])
    dist.init_process_group(backend=backend)


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def _get_free_gpus(
        threshold_vram_free: int = 4096,
        max_gpus: int = 1,
        wait: bool = False,
        sleep_time: int = 10
) -> List[int]:
    """
    Gets free gpus. Borrowed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696

    Args:
        threshold_vram_free (int, optional): A GPU is considered free if the vram usage is no less than
            the threshold. Defaults to 2048 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign. Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True.
            Default 10.

    Returns:
        A list of gpus with the largest available memory.
    """

    def _check():
        smi_query_result = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU", shell=True)
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        total_mem = list(filter(lambda info: "Total" in info, gpu_info))
        total_mem = [int(x.split(":")[1].replace("MiB", "").strip()) for x in total_mem]
        used_mem = list(filter(lambda info: "Used" in info, gpu_info))
        used_mem = [int(x.split(":")[1].replace("MiB", "").strip()) for x in used_mem]
        free_mem = [total - used for total, used in zip(total_mem, used_mem)]
        available = [(i, mem) for i, mem in enumerate(free_mem) if mem >= threshold_vram_free]
        free_gpus = [a[0] for a in sorted(available, key=lambda k: k[1])]
        free_gpus = free_gpus[: max(max_gpus, len(free_gpus))]
        return free_gpus

    while True:
        gpus_to_use = _check()
        if gpus_to_use or not wait:
            break
        logger.info(f"No free GPUs found, retrying in {sleep_time}s")
        time.sleep(sleep_time)

    if not gpus_to_use:
        raise RuntimeError("No free GPUs found")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in gpus_to_use])

    return gpus_to_use
