import math

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, _LRScheduler  # noqa
from torch.optim.optimizer import required  # noqa

from elephant.utils.logging_utils import get_logger

logger = get_logger("elephant")


class SGDW(torch.optim.Optimizer):
    """
    Implements stochastic gradient descent (optionally with momentum) with weight decay from the paper
    `Fixing Weight Decay Regularization in Adam`_. Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay factor (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(
        self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}.")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}.")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}.")

        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening.")

        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the models and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if weight_decay != 0:
                    p.data.add_(-weight_decay, p.data)

                p.data.add_(-group["lr"], d_p)

        return loss


class ExpAnnealLR(_LRScheduler):
    """
    Exponentially anneal the learning rate of each parameter group from the initial lr to end_lr over a number
    of iterations.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        end_lr (float): The final learning rate.
        iterations (int): The number of iterations over which to increase the learning rate.
        last_epoch (int): The index of the last iteration. Default: -1.
    """
    def __init__(self, optimizer, end_lr, iterations, last_epoch=-1):
        self.end_lr = end_lr
        self.iterations = iterations
        super(ExpAnnealLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        iteration = self.last_epoch + 1
        pct = iteration / self.iterations
        return [base_lr * (self.end_lr / base_lr) ** pct for base_lr in self.base_lrs]


class LinearSchedulerWithWarmup(LambdaLR):
    """
    Linearly increase the learning from 0 to initial learning rate during warmup and decrease the learning rate
    to 0 after the warmup. Uses LambaLR scheduler where the learning rate is multiplied by a lambda factor after
    calling scheduler.step().

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_train_steps (int): total number of training steps (number of batches * epochs).
        num_warmup_steps (int): number of training steps for learning rate warmup.
        last_epoch (int): The index of the last iteration. Default: -1. The scheduler will simply restart when
            resuming training from a checkpoint.
    """
    def __init__(self, optimizer, num_train_steps, num_warmup_steps, last_epoch=-1):
        def linear_lr_lambda(current_step: int):
            lambda_during_warmup = float(current_step) / float(max(1, num_warmup_steps))
            lambda_after_warmup = max(
                0.0,
                float(num_train_steps - current_step) / float(max(1, num_train_steps - num_warmup_steps)),
            )
            if current_step < num_warmup_steps:
                return lambda_during_warmup
            return lambda_after_warmup

        super(LinearSchedulerWithWarmup, self).__init__(optimizer, lr_lambda=linear_lr_lambda, last_epoch=last_epoch)


class ReduceLRWDOnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Reduce learning rate and weight decay when a metric has stopped improving. Models often benefit from reducing
    the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metric quantity and if
    no improvement is seen for a 'patience' number of epochs, the learning rate and weight decay factor is reduced
    for optimizers that implement the weight decay method from the paper `Fixing Weight Decay Regularization in Adam`.
    """

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1  # noqa
        self.last_epoch = epoch  # noqa

        if self.is_better(current, self.best):  # noqa
            self.best = current  # noqa
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1  # noqa
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:  # noqa
            self._reduce_lr(epoch)  # noqa
            self._reduce_weight_decay(epoch)
            self.cooldown_counter = self.cooldown  # noqa
            self.num_bad_epochs = 0  # noqa

    def _reduce_weight_decay(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):  # noqa
            if param_group["weight_decay"] != 0:
                old_weight_decay = float(param_group["weight_decay"])
                new_weight_decay = max(old_weight_decay * self.factor, self.min_lrs[i])  # noqa
                if old_weight_decay - new_weight_decay > self.eps:  # noqa
                    param_group["weight_decay"] = new_weight_decay
                    if self.verbose:  # noqa
                        logger.info(
                            f"Epoch {epoch}: reducing weight decay factor of group {i} to {new_weight_decay:.4e}."
                        )


class AnnealOnPlateau(object):
    """
    This class is a modification of torch.optim.lr_scheduler.ReduceLROnPlateau that enables setting an
    "auxiliary metric" to break ties. Reduce learning rate when a metric has stopped improving. Models
    often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler
    reads a metrics quantity and if no improvement is seen for a 'patience' number of epochs, the learning
    rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
    """

    def __init__(
        self,
        optimizer,
        mode="min",
        aux_mode="min",
        factor=0.1,
        patience=10,
        initial_extra_patience=0,
        verbose=False,
        cooldown=0,
        min_lr=0,
        eps=1e-8,
    ):

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an torch.optim.Optimizer")
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.default_patience = patience
        self.effective_patience = patience + initial_extra_patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.aux_mode = aux_mode
        self.best = None
        self.best_aux = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metric, auxiliary_metric=None):
        current = float(metric)
        epoch = self.last_epoch + 1
        self.last_epoch = epoch

        is_better = False

        if self.mode == "min":
            if current < self.best:
                is_better = True

        if self.mode == "max":
            if current > self.best:
                is_better = True

        if current == self.best and auxiliary_metric:
            current_aux = float(auxiliary_metric)
            if self.aux_mode == "min":
                if current_aux < self.best_aux:
                    is_better = True

            if self.aux_mode == "max":
                if current_aux > self.best_aux:
                    is_better = True

        if is_better:
            self.best = current
            if auxiliary_metric:
                self.best_aux = auxiliary_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.effective_patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            self.effective_patience = self.default_patience

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]  # noqa

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    logger.info(f"Epoch {epoch:5d}: reducing learning rate of group {i} to {new_lr:.4e}.")

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _init_is_better(self, mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")

        if mode == "min":
            self.mode_worse = math.inf
        else:  # mode == "max":
            self.mode_worse = -math.inf

        self.mode = mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode)
