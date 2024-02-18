import math
import copy
import sys
import os
import json
import time
from datetime import datetime
import random
import inspect
from pathlib import Path
from typing import List, Set, Dict, Any, Optional, Union, cast

import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR  # noqa
from torch.utils.data.sampler import Sampler as TorchSampler
import transformers

import elephant
from elephant.data.corpus import Corpus
from elephant.data.producer import DataProducer
from elephant.model import Model
from elephant.data import Dataset, DataLoader, DataIterator, MetaLangDataset
from elephant.utils.logging_utils import get_logger, log_line, add_file_handler

from ..optimizer import LinearSchedulerWithWarmup

logger = get_logger("elephant")


class ProbeTrainer(object):
    def __init__(self, corpus: Corpus, producer: DataProducer, model: Model):
        self.model = model
        self.corpus = corpus
        self.producer = producer
        self.trainer_cfg = elephant.config.trainer

        self.dataset = None
        self.checkpoint_path = None
        self.log_path = None

        self.log_to_file = self.trainer_cfg.get("log_to_file", False)
        self.train_log_file = self.trainer_cfg.get("train_log_file", "train.log")
        self.log_loss = self.trainer_cfg.get("log_loss", True)
        self.loss_log_file = self.trainer_cfg.get("loss_log_file", "loss.log")
        self.log_k_times = self.trainer_cfg.get("log_k_times", 10)
        self.save_optimizer_state = self.trainer_cfg.get("save_optimizer_state", False)
        self.save_model_each_k_epochs = self.trainer_cfg.get("save_model_each_k_epochs", 0)
        self.checkpoint = self.trainer_cfg.get("checkpoint", False)
        self.save_final_model = self.trainer_cfg.get("save_final_model", True)

        self.mini_batch_size = self.trainer_cfg.get("mini_batch_size", 8)
        self.eval_batch_size = self.trainer_cfg.get("eval_batch_size", None)

        self.min_learning_rate = self.trainer_cfg.get("min_learning_rate", 1.0e-7)

        self.max_epochs = self.trainer_cfg.get("max_epochs_probe", 30)

        self.train_with_dev = self.trainer_cfg.get("train_with_dev", False)
        self.shuffle = self.trainer_cfg.get("shuffle", True)

        self.eval_on_train_shuffle = self.trainer_cfg.get("eval_on_train_shuffle", False)
        self.main_evaluation_metric = self.trainer_cfg.get("main_evaluation_metric", "accuracy")
        self.use_final_model_for_eval = self.trainer_cfg.get("use_final_model_for_eval", False)

        self.anneal_with_restarts = self.trainer_cfg.get("anneal_with_restarts", False)
        self.anneal_with_prestarts = self.trainer_cfg.get("anneal_with_prestarts", False)
        self.anneal_against_dev_loss = False

        self.num_workers = self.trainer_cfg.get("num_workers", None)

        self.optimizer_cfg = self.trainer_cfg.optimizer
        self.scheduler_cfg = self.trainer_cfg.scheduler

        self.s_lang = None
        self.s_protos_dict = None

    def train(
            self,
            s_lang: str,
            epoch: int = 0,
    ) -> Dict:
        """
        Train any class that implements the elephant.models.MotelTemplate interface.
        """
        # ----------------------------------------------------------------------------------------------------
        # Initialize dataset and output directory
        # ----------------------------------------------------------------------------------------------------
        s_dataset_cfg = {
            "langs": [s_lang],
            "genres": {
                s_lang: elephant.config.source_langs.criteria.genres[s_lang]
            }
        }
        self.dataset = Dataset(self.producer, **s_dataset_cfg)
        self.checkpoint_path = elephant.checkpoint_path / self.trainer_cfg.objective / f"{s_lang}_probe"
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.log_path = elephant.log_path / self.trainer_cfg.objective / f"{s_lang}_probe"
        self.log_path.mkdir(exist_ok=True, parents=True)

        self.train_log_file = self.log_path / Path(self.train_log_file).name
        self.loss_log_file = self.log_path / Path(self.loss_log_file).name
        self.s_lang = s_lang

        probe_rank = elephant.config.model.task.probe_module.proj_dim
        result_dict = {
            f"val_{self.trainer_cfg.task}_acc_{probe_rank}": 0.0,
            f"test_{self.trainer_cfg.task}_acc_{probe_rank}": 0.0,
            f"val_{self.trainer_cfg.task}_confmat_{probe_rank}": None,
            f"test_{self.trainer_cfg.task}_confmat_{probe_rank}": None,
        }

        # ----------------------------------------------------------------------------------------------------
        # Validate parameters and check preconditions
        # ----------------------------------------------------------------------------------------------------

        src_checkpoint_path = elephant.checkpoint_path / self.trainer_cfg.objective / s_lang
        assert self.dataset.train, "Check training data."
        assert (src_checkpoint_path / "best_model.pt").is_file() or elephant.config.trainer.predict_mode == "pretrain",\
            "Check whether best_model fine-tuned on the source language has been saved."

        # ----------------------------------------------------------------------------------------------------
        # Prepare training environment and parameters
        # ----------------------------------------------------------------------------------------------------
        model_card: Dict[str, Any] = {
            "elephant_version": elephant.__version__,
            "pytorch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        local_variables = locals()
        training_parameters = {}
        for param in inspect.signature(self.train).parameters:
            training_parameters[param] = local_variables[param]
        for param in self.trainer_cfg:
            training_parameters[param] = self.trainer_cfg[param]
        model_card["training_parameters"] = training_parameters

        self.model.model_card = model_card

        # check for and delete previous best models
        self._check_for_and_delete_previous_best_models()

        # load the best model fine-tuned on the source language
        if elephant.config.trainer.predict_mode != "pretrain":
            self.model.load_state_dict(
                self.model.load(src_checkpoint_path / "best_model.pt").state_dict(), strict=True
            )
        self.model.to(elephant.device)

        # prepare loss logging file and set up header
        if self.log_loss and self.loss_log_file:
            open(self.loss_log_file, mode='w', encoding="utf-8").close()
        else:
            self.loss_log_file = None

        # set train and evaluation batch size
        if self.eval_batch_size is None:
            self.eval_batch_size = self.mini_batch_size

        # ----------------------------------------------------------------------------------------------------
        # Build optimizer
        # ----------------------------------------------------------------------------------------------------
        optimizer = self._build_optimizer(self.optimizer_cfg)

        initial_learning_rate, min_learning_rate = self.get_init_min_lr(
            optimizer, self.min_learning_rate
        )

        optimizer = cast(torch.optim.Optimizer, optimizer)

        # ----------------------------------------------------------------------------------------------------
        # Build scheduler
        # ----------------------------------------------------------------------------------------------------
        anneal_mode = "min" if self.train_with_dev or self.anneal_against_dev_loss else "max"
        best_validation_score = math.inf if self.train_with_dev or self.anneal_against_dev_loss else -1.0

        dataset_size = len(self.dataset.train)  # use source dataset size
        if self.train_with_dev:
            dataset_size += len(self.dataset.dev)

        scheduler = self._build_scheduler(
            scheduler_name=self.scheduler_cfg.name, optimizer=optimizer,
            initial_learning_rate=initial_learning_rate,
            epochs_tot=self.max_epochs,
            dataset_size=dataset_size,
            anneal_mode=anneal_mode,
        )

        log_bad_epochs = False

        # ----------------------------------------------------------------------------------------------------
        # Prepare training data
        # ----------------------------------------------------------------------------------------------------
        train_data = self.dataset.train

        if self.train_with_dev:
            parts = [self.dataset.train]
            if self.train_with_dev and self.dataset.dev:
                parts.append(self.dataset.dev)
            train_data = torch.utils.data.ConcatDataset(parts)

        # ----------------------------------------------------------------------------------------------------
        # Prepare evaluation data setting
        # ----------------------------------------------------------------------------------------------------
        # whether using dev dataset to evaluate
        if not self.train_with_dev and self.dataset.dev:
            log_dev = True
        else:
            log_dev = False

        # ----------------------------------------------------------------------------------------------------
        # Build data sampler
        # ----------------------------------------------------------------------------------------------------
        sampler = self._build_sampler(train_data)

        # ----------------------------------------------------------------------------------------------------
        # Start training
        # ----------------------------------------------------------------------------------------------------
        if epoch >= self.max_epochs:
            logger.warning(
                f"Starting at epoch {epoch + 1}/{self.max_epochs}. No training will be done."
            )

        train_loss_history = []

        dev_loss_history = []
        dev_score_history = []

        # at any point you can hit Ctrl + C to break out of training early.
        try:
            if self.log_to_file:
                log_handler = add_file_handler(logger, self.log_path / self.train_log_file)
            else:
                log_handler = None

            lr_info = ",".join([f"{lr:.8f}" for lr in initial_learning_rate])

            log_line(logger)
            logger.info(f'Model: "{self.model.model_cfg.name}"')
            log_line(logger)
            logger.info(f'Corpus: "{self.corpus.corpus_cfg.name}"')
            log_line(logger)
            logger.info("Parameters:")
            logger.info(f' - learning_rate: "{lr_info}"')
            logger.info(f' - mini_batch_size: "{self.mini_batch_size}"')
            logger.info(f' - patience: "{self.scheduler_cfg.patience}"')
            logger.info(f' - anneal_factor: "{self.scheduler_cfg.anneal_factor}"')
            logger.info(f' - max_epochs: "{self.max_epochs}"')
            logger.info(f' - shuffle: "{self.shuffle}"')
            log_line(logger)
            logger.info(f'Model training base path: "{elephant.output_root}"')
            logger.info(f'- checkpoint path: "{self.checkpoint_path}"')
            logger.info(f'- log path: "{self.log_path}"')
            log_line(logger)
            logger.info(f'Probe Module:')
            logger.info(f'- probe rank: "{probe_rank}"')
            log_line(logger)
            logger.info(f"Device: {elephant.device}")

            log_line(logger)
            logger.info("--- Training ---")

            # ------------------------------------------------------------------------------------------------
            # Loop epochs
            # ------------------------------------------------------------------------------------------------
            previous_learning_rate = initial_learning_rate
            momentum = [group["momentum"] if "momentum" in group else 0 for group in optimizer.param_groups]

            self.model.zero_grad(set_to_none=True)
            self.set_requires_grad(self.model.encoder, requires_grad=False)
            self.set_requires_grad(self.model.parser, requires_grad=False)
            self.set_requires_grad(self.model.probe_module, requires_grad=True)
            self.set_requires_grad(self.model.proto_module, requires_grad=False)

            s_dataloader = DataLoader(
                self.dataset.train,
                batch_size=self.mini_batch_size,
                shuffle=False,
                num_workers=0 if self.num_workers is None else self.num_workers,
                collate_fn=self.producer.collate_data,
                drop_last=False
            )
            self.model.eval()
            self.s_protos_dict = self.model.get_s_protos(dataloader=s_dataloader)

            for epoch in range(epoch + 1, self.max_epochs + 1):
                log_line(logger)

                # --------------------------------------------------------------------------------------------
                # Prepare variables before loop batches
                # --------------------------------------------------------------------------------------------
                # reset current learning rate
                current_learning_rate = self.reset_current_lr(optimizer)

                # stop training if learning rate becomes too small
                all_lrs_too_small = all([
                    lr < min_lr for lr, min_lr in zip(current_learning_rate, min_learning_rate)
                ])

                if not isinstance(scheduler, (OneCycleLR, LinearSchedulerWithWarmup)) \
                        and all_lrs_too_small:
                    log_line(logger)
                    logger.info("Learning rate too small, quitting training!")
                    log_line(logger)
                    break

                dataloader = DataLoader(
                    train_data,
                    batch_size=self.mini_batch_size,
                    shuffle=self.shuffle if epoch > 1 else False,  # never shuffle the first epoch
                    sampler=sampler,
                    num_workers=0 if self.num_workers is None else self.num_workers,
                    collate_fn=self.producer.collate_data,
                    drop_last=True
                )

                # --------------------------------------------------------------------------------------------
                # Reserve model states before running batches
                # --------------------------------------------------------------------------------------------
                model_card["training_parameters"]["epoch"] = epoch

                previous_learning_rate = current_learning_rate

                # --------------------------------------------------------------------------------------------
                # Start a training epoch
                # --------------------------------------------------------------------------------------------

                self.model.probe_module.train()
                # self.model.train()

                train_loss: float = 0
                seen_batches = 0
                total_number_of_batches = len(dataloader)
                modulo = max(1, int(total_number_of_batches / self.log_k_times))

                batch_time = 0.0
                average_over = 0

                data_iter = iter(dataloader)
                num_iterator = len(dataloader)

                for batch_no in range(0, num_iterator):
                    batch = next(data_iter)

                    start_time = time.time()

                    optimizer.zero_grad()
                    self.model.zero_grad(set_to_none=True)

                    loss_result, count = self.model.forward_loss_probe(batch=batch, protos_dict=self.s_protos_dict)

                    loss = loss_result.metric_score
                    loss.backward()

                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    optimizer.step()

                    average_over += count
                    train_loss += loss.item() * count

                    if isinstance(scheduler, (OneCycleLR, LinearSchedulerWithWarmup)):
                        scheduler.step()
                        current_learning_rate = self.reset_current_lr(optimizer)
                        momentum = self.reset_momentum(optimizer)

                    # log training info each modulo iterations
                    seen_batches += 1
                    batch_time += time.time() - start_time
                    if seen_batches % modulo == 0:
                        momentum_info = ""
                        if self.scheduler_cfg.cycle_momentum:
                            momentum_info = " - momentum:" + ",".join([f"{m:.4f}" for m in momentum])

                        lr_info = ",".join([f"{lr:.8f}" for lr in current_learning_rate])

                        if average_over > 0:
                            intermittent_loss = train_loss / average_over
                        else:
                            intermittent_loss = train_loss / seen_batches

                        logger.info(
                            f"epoch: {epoch}"
                            f" - iter: {seen_batches}/{total_number_of_batches}"
                            f" - loss: {intermittent_loss:.8f}"
                            f" - time (sec): {(time.time() - start_time):.2f}"
                            f" - samples/sec: {average_over / (time.time() - start_time):.2f}"
                            f" - lr: {lr_info}{momentum_info}"
                        )
                        logger.info(loss_result.log_header)
                        logger.info(loss_result.log_line)

                        batch_time = 0.0

                if average_over != 0:
                    train_loss /= average_over

                # --------------------------------------------------------------------------------------------
                # Start a training evaluation
                # --------------------------------------------------------------------------------------------
                self.model.eval()
                # save model each K epochs
                if self.save_model_each_k_epochs > 0 and epoch % self.save_model_each_k_epochs == 0:
                    logger.info("Saving model trained after current epoch.")
                    model_name = "model_epoch_" + str(epoch) + ".pt"
                    self.model.save(self.checkpoint_path / model_name, checkpoint=self.save_optimizer_state)

                log_line(logger)
                logger.info(f"EPOCH {epoch} done: loss {train_loss:.4f} - lr {lr_info}.")

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_dev:
                    assert self.dataset.dev
                    eval_dataloader = DataLoader(
                        dataset=self.dataset.dev,
                        batch_size=self.eval_batch_size,
                        num_workers=self.num_workers,  # noqa
                        collate_fn=self.producer.collate_data,
                        shuffle=False,
                        drop_last=False
                    )

                    dev_metric_result, dev_loss_result = self.model.evaluate_probe(
                        dataloader=eval_dataloader, protos_dict=self.s_protos_dict
                    )
                    result_line += f"\t{dev_metric_result.log_line}"
                    result_line += f"\t{dev_loss_result.log_line}"

                    logger.info(f"DEV")
                    logger.info(f"{dev_metric_result.log_header}")
                    logger.info(f"{dev_metric_result.log_line}")
                    logger.info(f"{dev_loss_result.log_header}")
                    logger.info(f"{dev_loss_result.log_line}")

                    dev_score = dev_metric_result.metric_score
                    dev_loss = dev_loss_result.metric_score

                    dev_score_history.append(dev_score)
                    dev_loss_history.append(dev_loss)

                # determine if this is the best model or if we need to anneal
                current_epoch_has_best_model_so_far = False

                # default mode: anneal against dev score
                if log_dev and not self.anneal_against_dev_loss:
                    if dev_score > best_validation_score:  # noqa
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_score
                        result_dict[f"val_{self.trainer_cfg.task}_acc_{probe_rank}"] = \
                            dev_metric_result.metric_detail[f"{self.trainer_cfg.task}_acc"]
                        result_dict[f"val_{self.trainer_cfg.task}_confmat_{probe_rank}"] = \
                            dev_metric_result.metric_detail[f"{self.trainer_cfg.task}_confusion_matrix"]

                # alternative: anneal against dev loss
                if log_dev and self.anneal_against_dev_loss:
                    if dev_loss < best_validation_score:  # noqa
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_loss

                # alternative: anneal against train loss
                if self.train_with_dev:
                    if train_loss < best_validation_score:
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = train_loss

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = scheduler.num_bad_epochs

                except AttributeError:
                    bad_epochs = 0

                new_learning_rate = self.reset_current_lr(optimizer)

                if any([
                    new_lr != prev_lr for new_lr, prev_lr in zip(new_learning_rate, previous_learning_rate)
                ]):
                    bad_epochs = self.scheduler_cfg.patience + 1
                    # lr unchanged
                    if all([
                        prev_lr == initial_lr
                        for prev_lr, initial_lr in zip(previous_learning_rate, initial_learning_rate)
                    ]):
                        bad_epochs += self.scheduler_cfg.initial_extra_patience

                # log bad epochs
                if log_bad_epochs:
                    logger.info(f"BAD EPOCHS (no improvement): {bad_epochs}")  # noqa

                if self.loss_log_file is not None:
                    with open(self.loss_log_file, mode='a') as f:
                        # make headers on first epoch
                        if epoch == 1:
                            bad_epoch_header = "BAD_EPOCHS\t" if log_bad_epochs else ""
                            f.write(f"EPOCH\tTIMESTAMP\t{bad_epoch_header}LEARNING_RATE\tTRAIN_LOSS")

                        lr_info = ",".join([f"{lr:.8f}" for lr in current_learning_rate])

                        bad_epoch_info = "\t" + str(bad_epochs) if log_bad_epochs else ""  # noqa
                        f.write(
                            f"\n{epoch}\t{datetime.now():%H:%M:%S}{bad_epoch_info}\t{lr_info}\t{train_loss}"
                        )
                        f.write(result_line)

                # if checkpoint is enabled, save model at each epoch
                if self.checkpoint:
                    self.model.save(self.checkpoint_path / "checkpoint.pt", checkpoint=True)

                # check whether to save best model
                if (
                        current_epoch_has_best_model_so_far  # noqa
                        and not self.use_final_model_for_eval
                ):
                    # not self.train_with_dev or self.anneal_with_restarts or self.anneal_with_prestarts
                    logger.info("Saving the best model.")
                    self.model.save(self.checkpoint_path / "best_model.pt", checkpoint=self.save_optimizer_state)

            # if not in parameter selection mode, save final model
            if self.save_final_model:
                self.model.save(self.checkpoint_path / "final_model.pt", checkpoint=self.save_optimizer_state)

        except KeyboardInterrupt:
            log_line(logger)
            logger.info("Exiting from training early.")

            logger.info("Saving models ...")
            self.model.save(
                self.checkpoint_path / "final_model.pt", checkpoint=self.save_optimizer_state
            )
            logger.info("Done.")

        except Exception as error:
            if self.log_to_file:
                log_handler.close()  # noqa
                logger.removeHandler(log_handler)
            raise error

        finally:
            pass

        # save projected protos and trained projection
        self.save_protos_and_proj(checkpoint_path=self.checkpoint_path)

        # test best models if test data is present
        if self.dataset.test:
            final_score, test_metric_result = self.final_test(checkpoint_path=self.checkpoint_path)
            result_dict[f"test_{self.trainer_cfg.task}_acc_{probe_rank}"] = \
                test_metric_result.metric_detail[f"{self.trainer_cfg.task}_acc"]
            result_dict[f"test_{self.trainer_cfg.task}_confmat_{probe_rank}"] = \
                test_metric_result.metric_detail[f"{self.trainer_cfg.task}_confusion_matrix"]
            with open(
                    self.log_path / f"results_probe_{self.trainer_cfg.predict_mode}.json", mode="a", encoding="utf-8"
            ) as fh:
                print(json.dumps(result_dict), file=fh)
        else:
            final_score = 0
            logger.info("Test data not provided, setting final score to 0.")

        if self.log_to_file:
            log_handler.close()  # noqa
            logger.removeHandler(log_handler)

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
            "result_dict": result_dict,
        }

    def save_protos_and_proj(self, checkpoint_path: Union[Path, str]):
        if (checkpoint_path / "best_model.pt").exists():
            self.model.load_state_dict(self.model.load(checkpoint_path / "best_model.pt").state_dict())
        else:
            logger.info("Using last state of model ...")

        self.model.eval()
        xs_protos = self.s_protos_dict["xs_proto"]
        ys_protos = self.s_protos_dict["ys_proto"]
        zs_protos = self.model.probe_module.proj(xs_protos)
        proto_dict = {
            "proj_protos": zs_protos.detach().cpu(),
            "protos": xs_protos.detach().cpu(),
            "y_protos": ys_protos.detach().cpu(),
        }
        proto_data_dir = elephant.output_root / "protos" / self.trainer_cfg.objective
        proto_data_dir.mkdir(exist_ok=True, parents=True)
        torch.save(
            proto_dict, proto_data_dir / f"{self.s_lang}_protos_{elephant.config.model.task.probe_module.proj_dim}.pt"
        )

        # save trained projection
        projection = self.model.probe_module.proj.proj.data.detach().cpu()
        projection_dir = elephant.output_root / "protos_proj" / self.trainer_cfg.objective
        projection_dir.mkdir(exist_ok=True, parents=True)
        torch.save(
            projection, projection_dir / f"{self.s_lang}_protos_{elephant.config.model.task.probe_module.proj_dim}.pt"
        )

    def final_test(
            self,
            checkpoint_path: Union[Path, str],
    ):
        log_line(logger=logger)

        self.model.eval()

        if (checkpoint_path / "best_model.pt").exists():
            self.model.load_state_dict(self.model.load(checkpoint_path / "best_model.pt").state_dict())
        else:
            logger.info("Testing using last state of model ...")

        assert self.dataset.test
        test_dataloader = DataLoader(
            dataset=self.dataset.test,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,  # noqa
            collate_fn=self.producer.collate_data,
            shuffle=False,
            drop_last=False
        )
        test_metric_result, test_loss_result = self.model.evaluate_probe(
            dataloader=test_dataloader, protos_dict=self.s_protos_dict
        )
        logger.info(f"TEST")
        logger.info(test_metric_result.log_header)
        logger.info(test_metric_result.log_line)
        log_line(logger=logger)
        final_score = test_metric_result.metric_score

        return final_score, test_metric_result

    def _check_for_and_delete_previous_best_models(self):
        best_models = [f for f in self.checkpoint_path.glob("best_model*") if f.is_file()]
        for model in best_models:
            model.unlink()

    def _build_scheduler(
            self,
            scheduler_name: str,
            optimizer: torch.optim.Optimizer,
            initial_learning_rate,
            epochs_tot: int,
            dataset_size: int,
            anneal_mode: str,
    ):
        scheduler = self.get_scheduler(
            scheduler_name=scheduler_name, dataset_size=dataset_size, batch_size=self.mini_batch_size,
            optimizer=optimizer, initial_learning_rate=initial_learning_rate,
            epochs_tot=epochs_tot, warmup_fraction=self.scheduler_cfg.warmup_fraction,
            cycle_momentum=self.scheduler_cfg.cycle_momentum, anneal_factor=self.scheduler_cfg.anneal_factor,
            anneal_mode=anneal_mode, patience=self.scheduler_cfg.patience,
            initial_extra_patience=self.scheduler_cfg.initial_extra_patience
        )
        return scheduler

    def _build_optimizer(self, optimizer_cfg) -> torch.optim.Optimizer:
        if optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.probe_module.parameters(),
                lr=optimizer_cfg.lr_probe,
                betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
                weight_decay=optimizer_cfg.weight_decay_probe
            )
        elif optimizer_cfg.name == "SGD":
            optimizer = torch.optim.SGD(
                self.model.probe_module.parameters(),
                lr=optimizer_cfg.lr_probe,
                weight_decay=optimizer_cfg.weight_decay_probe,
                momentum=optimizer_cfg.momentum
            )
        else:
            raise NotImplementedError(f'Optimizer "{optimizer_cfg.name}" is not implemented.')
        return optimizer

    @staticmethod
    def _build_sampler(dataset: DataIterator) -> Optional[TorchSampler]:
        if elephant.config.trainer.dist:
            sampler = TorchSampler(dataset)
        else:
            sampler = None

        return sampler

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    @staticmethod
    def get_init_min_lr(optimizer: torch.optim.Optimizer, min_learning_rate: float):
        initial_learning_rate = [group["lr"] for group in optimizer.param_groups]
        min_learning_rate = [min_learning_rate] * len(initial_learning_rate)
        for i, lr in enumerate(initial_learning_rate):
            if lr < min_learning_rate[i]:
                min_learning_rate[i] = lr / 10
        return initial_learning_rate, min_learning_rate

    @staticmethod
    def get_scheduler(
            scheduler_name: str,
            dataset_size: int,
            batch_size: int,
            optimizer: torch.optim.Optimizer,
            initial_learning_rate,
            epochs_tot: int,
            warmup_fraction: float = -1,
            cycle_momentum: bool = False,
            anneal_factor: float = 0.5,
            anneal_mode: str = "max",
            patience: int = 3,
            initial_extra_patience: int = 0,
    ):
        if scheduler_name == "OneCycleLR":
            steps_per_epoch = int((dataset_size + batch_size - 1) / batch_size)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=initial_learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs_tot,
                pct_start=0.0,
                cycle_momentum=cycle_momentum,
            )
        elif scheduler_name == "LinearSchedulerWithWarmup":
            steps_per_epoch = int((dataset_size + batch_size - 1) / batch_size)
            # num_train_steps = steps_per_epoch * self.max_epochs
            num_train_steps = steps_per_epoch * epochs_tot
            num_warmup_steps = num_train_steps * warmup_fraction if warmup_fraction > 0 else 1
            scheduler = LinearSchedulerWithWarmup(
                optimizer,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            raise NotImplementedError(f"Scheduler {scheduler_name} not implemented.")

        return scheduler

    @staticmethod
    def get_init_momentum(optimizer: torch.optim.Optimizer):
        return [group["momentum"] if "momentum" in group else 0 for group in optimizer.param_groups]

    @staticmethod
    def reset_current_lr(optimizer: torch.optim.Optimizer):
        return [group["lr"] for group in optimizer.param_groups]

    @staticmethod
    def reset_momentum(optimizer: torch.optim.Optimizer):
        return [group["betas"][0] if "betas" in group else group.get("momentum", 0)
                for group in optimizer.param_groups]

    @staticmethod
    def estimate_proportion(data_loader, n_clusters):
        y = torch.LongTensor().to(elephant.device)
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, dict):
                    target = batch["deprel_ids"].flatten().to(elephant.device)
                else:
                    assert isinstance(batch, tuple), f"Unsupported data type of batches in data_loader: {type(batch)}."
                    target = batch[1].flatten().to(elephant.device)
                y = torch.cat((y, target))
        nb_sample_s = torch.tensor([torch.sum(y == i) for i in range(n_clusters)]).float()
        nb_sample_s = torch.where(nb_sample_s == 0.0, torch.ones_like(nb_sample_s), nb_sample_s)
        # assert nb_sample_s[0] == 0.0, "There should be no tags belonging to `_`."
        proportion = nb_sample_s / torch.sum(nb_sample_s)
        return proportion
