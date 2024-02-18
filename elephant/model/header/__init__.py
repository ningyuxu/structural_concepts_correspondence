from .template import TaskModelTemplate as TaskModel
from .syntax_aligning import ProbeModel, LSAdaProtoNet, MetaProtoNet
from .dependency_parsing import DependencyParsingModel, RelModel
from .pos_tagging import POSModel


def get_task(task_cfg) -> TaskModel:
    task = {
        "dependency_parsing": DependencyParsingModel,
        "rel_prediction": RelModel,
        "pos_tagging": POSModel,
        "probe": ProbeModel,
        "meta_proto": MetaProtoNet,
        "ls_ada_proto": LSAdaProtoNet,
    }[task_cfg.name](task_cfg)
    return task


__all__ = [
    "get_task",
    "DependencyParsingModel",
    "RelModel",
    "POSModel",
    "ProbeModel",
    "MetaProtoNet",
    "LSAdaProtoNet",
]
