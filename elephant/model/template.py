from pathlib import Path
from typing import Dict, Tuple, Union

import torch

import elephant
from elephant.trainer.result import Result
from elephant.utils.logging_utils import get_logger

logger = get_logger("elephant")


class ModelTemplate(torch.nn.Module):
    def __init__(self, model_cfg):
        super(ModelTemplate, self).__init__()

        self.model_cfg = model_cfg
        self.model_card = None

    def forward_loss(self, **kwargs) -> Tuple[Result, int]:
        """
        Performs a forward pass and returns a loss tensor for backpropagation.
        """
        raise NotImplementedError

    def evaluate(self, **kwargs) -> Tuple[Result, Result]:
        raise NotImplementedError

    def save(self, model_file: Union[str, Path], checkpoint: bool = False) -> None:
        model_state = self._get_state_dict()

        # write out a "model card" if one is set
        if self.model_card is not None:
            if "training_parameters" in self.model_card:
                training_parameters = self.model_card["training_parameters"]

                if "optimizer" in training_parameters:
                    optimizer = training_parameters["optimizer"]
                    if checkpoint:
                        training_parameters["optimizer_state_dict"] = optimizer.state_dict()
                    training_parameters["optimizer"] = optimizer.__class__

                if "scheduler" in training_parameters:
                    scheduler = training_parameters["scheduler"]
                    if checkpoint:
                        training_parameters["scheduler_state_dict"] = scheduler.state_dict()
                    training_parameters["scheduler"] = scheduler.__class__

            model_state["model_card"] = self.model_card

        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

        # restore optimizer and scheduler to model card if set
        if self.model_card is not None:
            if optimizer:  # noqa
                self.model_card["training_parameters"]["optimizer"] = optimizer  # noqa
            if scheduler:  # noqa
                self.model_card["training_parameters"]["scheduler"] = scheduler  # noqa

    def load(self, model_file: Union[str, Path]):
        with open(model_file, mode="rb") as fh:
            model_state = torch.load(fh, map_location="cpu")

        model = self._init_model_with_state_dict(model_state, self.model_cfg)
        if "model_card" in model_state:
            model.model_card = model_state["model_card"]
        model.eval()
        model.to(elephant.device)

        return model

    def print_model_card(self):
        if hasattr(self, "model_card"):
            param_out = "\n--------------------------------------------------------------------------------\n"
            param_out += "--------------------------- Elephant Model Card --------------------------------\n"
            param_out += "--------------------------------------------------------------------------------\n"
            param_out += f"This Elephant model was trained with:\n"
            param_out += f'-- Elephant version {self.model_card["elephant_version"]}\n'
            param_out += f'-- PyTorch version {self.model_card["pytorch_version"]}\n'
            param_out += f'-- Transformers version {self.model_card["transformers_version"]}\n'
            param_out += "--------------------------------------------------------------------------------\n"

            param_out += "--------------------------- Training Parameters: -------------------------------\n"
            param_out += "--------------------------------------------------------------------------------\n"
            training_params = '\n'.join(
                f'-- {param} = {self.model_card["training_parameters"][param]}'
                for param in self.model_card["training_parameters"]
            )
            param_out += training_params + "\n"
            param_out += "--------------------------------------------------------------------------------\n"

            logger.info(param_out)
        else:
            logger.info("This model has no `Elephant Model Card`.")

    def _get_state_dict(self) -> Dict:
        state_dict = {"state_dict": self.state_dict()}

        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state: Dict, model_cfg):
        model = cls(model_cfg)
        model.load_state_dict(state["state_dict"])

        return model
