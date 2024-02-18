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
from elephant.data import Dataset, DataLoader, DataIterator, LSMetaLangDataset
from elephant.utils.logging_utils import get_logger, log_line, add_file_handler

from elephant.utils.predict_utils import HDF5Dataset

from ..optimizer import LinearSchedulerWithWarmup, AnnealOnPlateau

logger = get_logger("elephant")


class LSAdaProtoNetTrainer(object):
    def __init__(self, corpus: Corpus, producer: DataProducer, model: Model):
        self.model = model
        self.corpus = corpus
        self.producer = producer
        self.trainer_cfg = elephant.config.trainer

        self.s_dataset = None
        self.meta_dataset = None
        self.val_dataset = None
        self.t_dataset = None
        # self.meta_langs = None
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

        self.max_epochs = self.trainer_cfg.get("max_epochs_proto", 100)
        self.max_epochs_adaptor = self.trainer_cfg.get("max_epochs_adaptor", 30)

        self.train_with_dev = self.trainer_cfg.get("train_with_dev", False)
        self.shuffle = self.trainer_cfg.get("shuffle", True)

        self.eval_on_train_shuffle = self.trainer_cfg.get("eval_on_train_shuffle", False)
        self.main_evaluation_metric = self.trainer_cfg.get("main_evaluation_metric", "accuracy")
        self.use_final_model_for_eval = self.trainer_cfg.get("use_final_model_for_eval", False)

        self.anneal_with_restarts = self.trainer_cfg.get("anneal_with_restarts", False)
        self.anneal_with_prestarts = self.trainer_cfg.get("anneal_with_prestarts", False)
        self.anneal_against_dev_loss = self.trainer_cfg.get("anneal_against_dev_loss", False)

        self.num_workers = self.trainer_cfg.get("num_workers", None)

        self.optimizer_cfg = self.trainer_cfg.optimizer
        self.scheduler_cfg = self.trainer_cfg.scheduler

        self.valid_labels_record = None
        self.s_protos_dict = None

    def train(
            self,
            epoch: int = 0,
            meta_langs: List = None,
            val_langs: List = None,
            exp_cfg: Dict = None,
    ) -> Dict:
        """
        Train any class that implements the elephant.models.MotelTemplate interface.
        """
        # ----------------------------------------------------------------------------------------------------
        # Initialize dataset and output directory
        # ----------------------------------------------------------------------------------------------------
        if meta_langs is None:
            if "meta_langs" in exp_cfg:
                meta_langs = exp_cfg["meta_langs"]
            else:
                all_langs = set(elephant.config.source_langs.criteria.langs)
                tgt_langs = set(elephant.config.target_langs.criteria.langs)
                meta_langs = all_langs.difference(tgt_langs)
        meta_langs = list(meta_langs)
        # self.meta_langs = meta_langs

        if val_langs is None:
            if "val_langs" in exp_cfg:
                val_langs = exp_cfg["val_langs"]
            else:
                val_langs = elephant.config.val_langs.criteria.langs
        val_langs = list(val_langs)

        s_lang = exp_cfg["src_lang"]
        t_lang = exp_cfg["tgt_lang"]
        k_shots = self.trainer_cfg.ns

        s_dataset_cfg = {
            "langs": [s_lang],
            "genres": {
                s_lang: elephant.config.source_langs.criteria.genres[s_lang]
            }
        }
        self.s_dataset = Dataset(self.producer, **s_dataset_cfg)

        meta_datasets = []
        samples_per_dataset = []
        for lang in meta_langs:
            dataset_cfg = {
                "langs": [lang], "genres": {lang: elephant.config.source_langs.criteria.genres[lang]}
            }
            dataset = Dataset(self.producer, **dataset_cfg)
            meta_datasets.append(dataset)
            samples_per_dataset.append(len(dataset.train))
        ave_num_samples = np.mean(samples_per_dataset).item()
        self.meta_dataset = LSMetaLangDataset(meta_datasets, task=self.trainer_cfg.task)

        val_datasets = []
        for lang in val_langs:
            dataset_cfg = {
                "langs": [lang], "genres": {lang: elephant.config.source_langs.criteria.genres[lang]}
            }
            dataset = Dataset(self.producer, **dataset_cfg)
            val_datasets.append(dataset)
        self.val_dataset = LSMetaLangDataset(val_datasets, task=self.trainer_cfg.task)

        t_dataset_cfg = {
            "langs": [t_lang],
            "genres": {
                t_lang: elephant.config.source_langs.criteria.genres[t_lang]
            }
        }
        self.t_dataset = Dataset(self.producer, **t_dataset_cfg)
        self.checkpoint_path = elephant.checkpoint_path / self.trainer_cfg.objective / f"{s_lang}_{k_shots}"
        # self.checkpoint_path = elephant.checkpoint_path / self.trainer_cfg.objective / f"{k_shots}"
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.log_path = elephant.log_path / self.trainer_cfg.objective / f"{s_lang}_{k_shots}"
        # self.log_path = elephant.log_path / self.trainer_cfg.objective / f"{k_shots}"
        self.log_path.mkdir(exist_ok=True, parents=True)

        self.train_log_file = self.log_path / Path(self.train_log_file).name
        self.loss_log_file = self.log_path / Path(self.loss_log_file).name

        # ----------------------------------------------------------------------------------------------------
        # Validate parameters and check preconditions
        # ----------------------------------------------------------------------------------------------------
        src_checkpoint_path = elephant.checkpoint_path / self.trainer_cfg.objective / f"{s_lang}_probe"
        # assert self.s_dataset.train, "Check training data."
        assert (src_checkpoint_path / "best_model.pt").is_file(), \
            "Check whether best_model fine-tuned on the source language has been saved."

        all_langs = elephant.config.source_langs.criteria.langs
        self.valid_labels_record = self.get_valid_labels_lang_pair(s_lang=s_lang, langs=all_langs)

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
        self.model.load_state_dict(self.model.load(src_checkpoint_path / "best_model.pt").state_dict(), strict=True)

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

        scheduler = self._build_scheduler(
            scheduler_name=self.scheduler_cfg.name, optimizer=optimizer,
            initial_learning_rate=initial_learning_rate, epochs_tot=self.max_epochs,
            dataset_size=self.meta_dataset.num_meta_langs * self.trainer_cfg.dataset_per_lang,
            batch_size=1, anneal_mode=anneal_mode
        )

        log_bad_epochs = True if isinstance(scheduler, AnnealOnPlateau) else False

        # ----------------------------------------------------------------------------------------------------
        # Prepare evaluation data setting
        # ----------------------------------------------------------------------------------------------------
        # whether using dev dataset to evaluate
        if self.val_dataset is not None:
            log_dev = True
        else:
            log_dev = False

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
            logger.info(f'Proto Module:')
            logger.info(f'- hid_dim: "{elephant.config.model.task.proto_module.hid_dim}"')
            logger.info(f'- proj_dim: "{elephant.config.model.task.proto_module.proj_dim}"')
            logger.info(f'- dropout: "{elephant.config.model.task.proto_module.dropout}"')
            logger.info(f'- distance: "{elephant.config.model.task.proto_module.dist}"')
            log_line(logger)
            logger.info(f"Device: {elephant.device}")

            log_line(logger)
            logger.info("--- Meta Training ---")

            # ------------------------------------------------------------------------------------------------
            # Loop epochs
            # ------------------------------------------------------------------------------------------------
            previous_learning_rate = initial_learning_rate
            momentum = self.get_init_momentum(optimizer)

            self.set_requires_grad(self.model.encoder, requires_grad=False)
            self.set_requires_grad(self.model.parser, requires_grad=False)
            self.set_requires_grad(self.model.probe_module, requires_grad=False)
            self.set_requires_grad(self.model.proto_module, requires_grad=True)

            s_dataloader = DataLoader(
                self.s_dataset.train, batch_size=self.mini_batch_size, shuffle=False,
                num_workers=0 if self.num_workers is None else self.num_workers,
                collate_fn=self.producer.collate_data, drop_last=False
            )
            self.model.eval()
            self.s_protos_dict = self.model.get_s_protos(dataloader=s_dataloader, proj=True)

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

                meta_dataset, lang_ids = self.meta_dataset.load(
                    ns=self.trainer_cfg.ns, nq=self.trainer_cfg.nq, smooth=self.trainer_cfg.smooth,
                    dataset_per_lang=self.trainer_cfg.dataset_per_lang, split="train"
                )

                # --------------------------------------------------------------------------------------------
                # Reserve model states before running batches
                # --------------------------------------------------------------------------------------------
                model_card["training_parameters"]["epoch"] = epoch

                previous_learning_rate = current_learning_rate

                # --------------------------------------------------------------------------------------------
                # Start a training epoch
                # --------------------------------------------------------------------------------------------

                self.model.proto_module.train()
                # self.model.train()

                train_loss: float = 0
                seen_batches = 0
                total_number_of_batches = len(meta_dataset)
                modulo = max(1, int(total_number_of_batches / self.log_k_times))

                batch_time = 0.0
                average_over = 0

                num_iterator = len(meta_dataset)

                for task_no in range(0, num_iterator):
                    start_time = time.time()

                    optimizer.zero_grad()
                    self.model.zero_grad(set_to_none=True)

                    episode = meta_dataset[task_no]
                    support_dataloader = DataLoader(
                        episode["support"], batch_size=self.mini_batch_size, shuffle=False,
                        num_workers=0 if self.num_workers is None else self.num_workers,
                        collate_fn=self.producer.collate_data, drop_last=False
                    )
                    query_dataloader = DataLoader(
                        episode["query"], batch_size=self.mini_batch_size, shuffle=False,
                        num_workers=0 if self.num_workers is None else self.num_workers,
                        collate_fn=self.producer.collate_data, drop_last=False
                    )

                    t_lang = self.meta_dataset.id2lang[lang_ids[task_no]]

                    loss_result, count = self.model.forward_loss(
                        support_dataloader=support_dataloader, query_dataloader=query_dataloader,
                        s_protos_dict=self.s_protos_dict, idx=lang_ids[task_no],
                        valid_labels=self.valid_labels_record[f"{s_lang}_to_{t_lang}"]
                    )

                    loss = loss_result.metric_score
                    loss.backward()

                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    optimizer.step()
                    # self.model.orthogonalize_adaptor(meta_train=True, idx=lang_ids[task_no])

                    average_over += count
                    train_loss += loss.item() * count

                    # torch.cuda.empty_cache()

                    # do the scheduler step if one-cycle or linear decay
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
                    logger.info("Saving model of current epoch.")
                    model_name = "model_epoch_" + str(epoch) + ".pt"
                    self.model.save(self.checkpoint_path / model_name, checkpoint=self.save_optimizer_state)

                log_line(logger)
                logger.info(f"EPOCH {epoch} done: loss {train_loss:.4f} - lr {lr_info}.")

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_dev:
                    val_dataset, lang_ids = self.val_dataset.load(
                        ns=self.trainer_cfg.ns, nq=self.trainer_cfg.nq, smooth=self.trainer_cfg.smooth,
                        dataset_per_lang=self.trainer_cfg.eval_dataset_per_lang, split="train",
                    )
                    acc_record = []
                    loss_record = []

                    val_num_iterator = len(val_dataset)

                    for val_task_no in range(0, val_num_iterator):
                        val_task = val_dataset[val_task_no]
                        support_dataloader = DataLoader(
                            val_task["support"], batch_size=self.mini_batch_size, shuffle=False,
                            num_workers=0 if self.num_workers is None else self.num_workers,
                            collate_fn=self.producer.collate_data, drop_last=False
                        )
                        query_dataloader = DataLoader(
                            val_task["query"], batch_size=self.mini_batch_size, shuffle=False,
                            num_workers=0 if self.num_workers is None else self.num_workers,
                            collate_fn=self.producer.collate_data, drop_last=False
                        )
                        # self.train_adaptor(support_dataset=val_task["support"])

                        # assert (self.checkpoint_path / "best_adaptor.pt").is_file()
                        # self.model.load_adaptor(self.checkpoint_path / "best_adaptor.pt")

                        with torch.no_grad():
                            t_lang = self.val_dataset.id2lang[lang_ids[val_task_no]]
                            dev_loss_result, count = self.model.forward_loss(
                                support_dataloader=support_dataloader, query_dataloader=query_dataloader,
                                s_protos_dict=self.s_protos_dict, idx=lang_ids[val_task_no],
                                valid_labels=self.valid_labels_record[f"{s_lang}_to_{t_lang}"]
                            )
                        loss_record.append(dev_loss_result.metric_detail["loss_val"])
                        acc_record.append(dev_loss_result.metric_detail["acc_val"])

                        result_line += f"\t{dev_loss_result.log_line}"
                        logger.info(f"DEV")
                        logger.info(f"{dev_loss_result.log_header}")
                        logger.info(f"{dev_loss_result.log_line}")

                    dev_score = np.mean(acc_record).item()
                    dev_loss = np.mean(loss_record).item()

                    dev_loss_history.append(dev_score)

                # determine if this is the best model or if we need to anneal
                current_epoch_has_best_model_so_far = False

                # default mode: anneal against dev score
                if log_dev and not self.anneal_against_dev_loss:
                    if dev_score > best_validation_score:  # noqa
                        current_epoch_has_best_model_so_far = True
                        best_validation_score = dev_score

                # anneal against dev loss
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

        # test best models if test data is present
        if self.t_dataset.test:
            log_dir = elephant.log_path / self.trainer_cfg.objective / \
                      f"results-{self.trainer_cfg.task}-meta-proto-{k_shots}shots" / s_lang
            log_dir.mkdir(exist_ok=True, parents=True)
            final_scores = []
            rank = elephant.config.model.task.proto_module.proj_dim
            for lang in elephant.config.source_langs.criteria.langs:
                result_dict = {
                    f"{self.trainer_cfg.task}_acc_{rank}": 0.0, f"{self.trainer_cfg.task}_confmat_{rank}": None,
                }
                t_dataset_cfg = {
                    "langs": [lang],
                    "genres": {
                        lang: elephant.config.source_langs.criteria.genres[lang]
                    }
                }
                t_dataset = Dataset(self.producer, **t_dataset_cfg)
                score, test_metric_result = self.final_test(
                    checkpoint_path=self.checkpoint_path,  s_lang=s_lang, t_lang=lang, t_dataset=t_dataset
                )
                final_scores.append(score)

                try:
                    result_dict[f"{self.trainer_cfg.task}_acc_{rank}"] = test_metric_result.metric_detail[
                        f"{self.trainer_cfg.task}_acc"
                    ]
                    result_dict[f"{self.trainer_cfg.task}_confmat_{rank}"] = test_metric_result.metric_detail[
                        f"{self.trainer_cfg.task}_confusion_matrix"
                    ]
                except AttributeError:
                    result_dict[f"{elephant.config.trainer.task}_acc_{rank}"] = 0.0
                    result_dict[f"{elephant.config.trainer.task}_confmat_{rank}"] = None
                with open(log_dir / f"results_{lang}.json", mode='a', encoding="utf-8") as fh:
                    print(json.dumps(result_dict), file=fh)
            final_score = np.mean(final_scores).item()
            # final_score = self.final_test(checkpoint_path=self.checkpoint_path)
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
            "dev_loss_history": dev_loss_history
        }

    def train_adaptor(self, support_dataset):

        self.model.reset_adaptor_params()

        optimizer_adaptor = self._build_adaptor_optimizer(self.optimizer_cfg)
        adaptor_initial_learning_rate, adaptor_min_learning_rate = self.get_init_min_lr(
            optimizer_adaptor, self.min_learning_rate
        )
        optimizer_adaptor = cast(torch.optim.Optimizer, optimizer_adaptor)
        scheduler_adaptor = self._build_scheduler(
            scheduler_name=self.scheduler_cfg.name, optimizer=optimizer_adaptor,
            initial_learning_rate=adaptor_initial_learning_rate, epochs_tot=self.max_epochs_adaptor,
            dataset_size=len(support_dataset), batch_size=self.mini_batch_size, anneal_mode="min"
        )
        self.set_requires_grad(self.model.encoder, requires_grad=False)
        self.set_requires_grad(self.model.parser, requires_grad=False)
        self.set_requires_grad(self.model.probe_module, requires_grad=False)
        self.set_requires_grad(self.model.proto_module.proto_net, requires_grad=False)
        self.set_requires_grad(self.model.proto_module.ls_adaptors, requires_grad=False)
        self.set_requires_grad(self.model.proto_module.adaptor, requires_grad=True)
        self.model.proto_module.adaptor.train()

        adaptor_best_validation_score = math.inf

        for inner_epoch in range(0, self.max_epochs_adaptor):

            adaptor_train_loss: float = 0
            adaptor_average_over = 0

            ada_dataloader = DataLoader(
                support_dataset, batch_size=self.mini_batch_size, shuffle=True,
                num_workers=0 if self.num_workers is None else self.num_workers,
                collate_fn=self.producer.collate_data, drop_last=False
            )
            data_iter = iter(ada_dataloader)
            num_iterator = len(ada_dataloader)

            for batch_no in range(0, num_iterator):
                batch = next(data_iter)
                optimizer_adaptor.zero_grad()
                self.model.zero_grad(set_to_none=True)
                ada_loss_result, ada_count = self.model.forward_loss_adaptor(
                    batch=batch, s_protos_dict=self.s_protos_dict
                )
                loss_ada = ada_loss_result.metric_score
                loss_ada.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer_adaptor.step()
                # self.model.orthogonalize_adaptor(meta_train=False)

                adaptor_average_over += ada_count
                adaptor_train_loss += loss_ada.item() * ada_count

                # do the scheduler step if one-cycle or linear decay
                if isinstance(scheduler_adaptor, (OneCycleLR, LinearSchedulerWithWarmup)):
                    scheduler_adaptor.step()

            # logger.info(f" - Inner Epoch: {inner_epoch}")
            # logger.info(ada_loss_result.log_header)
            # logger.info(ada_loss_result.log_line)

            if adaptor_average_over != 0:
                adaptor_train_loss /= adaptor_average_over

            adaptor_current_epoch_has_best_model_so_far = False
            if adaptor_train_loss < adaptor_best_validation_score:
                adaptor_current_epoch_has_best_model_so_far = True
                adaptor_best_validation_score = adaptor_train_loss

            if adaptor_current_epoch_has_best_model_so_far:
                self.model.save_adaptor(self.checkpoint_path / "best_adaptor.pt")
        log_line(logger=logger)
        logger.info("Finish Adaptor Training.")
        log_line(logger=logger)

    def final_test(
            self,
            checkpoint_path: Union[Path, str],
            s_lang: str,
            t_lang: str,
            t_dataset: Dataset,
    ):
        log_line(logger=logger)

        if (checkpoint_path / "best_model.pt").exists():
            self.model.load_state_dict(self.model.load(checkpoint_path / "best_model.pt").state_dict())
        else:
            logger.info("Testing using last state of model ...")

        assert t_dataset.test
        use_test_as_support = False
        try:
            train_dataset = t_dataset.train
        except AssertionError:
            train_dataset = t_dataset.test
            use_test_as_support = True
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        support_ids = indices[:self.trainer_cfg.ns]
        if use_test_as_support:
            if self.trainer_cfg.ns >= len(train_dataset):
                d_query = None
            else:
                test_ids = indices[self.trainer_cfg.ns:]
                d_query = torch.utils.data.Subset(train_dataset, test_ids)
        d_support = torch.utils.data.Subset(train_dataset, support_ids)

        support_dataloader = DataLoader(
            dataset=d_support, batch_size=self.eval_batch_size, collate_fn=self.producer.collate_data,
            num_workers=0 if self.num_workers is None else self.num_workers,
            shuffle=False, drop_last=False
        )
        if (not use_test_as_support) or (d_query is not None):
            self.train_adaptor(support_dataset=d_support)
            assert (self.checkpoint_path / "best_adaptor.pt").is_file()
            self.model.load_adaptor(self.checkpoint_path / "best_adaptor.pt")

        self.model.eval()
        with torch.no_grad():
            support_set = self.model.get_support_set_for_eval(dataloader=support_dataloader)
            if use_test_as_support and d_query is None:
                final_score = 0.0
                test_metric_result = {
                    f"{self.trainer_cfg.task}_acc": 0.0,
                    f"{self.trainer_cfg.task}_confusion_matrix": None,
                }
            else:
                # Testing
                if use_test_as_support and d_query is not None:
                    test_dataloader = DataLoader(
                        dataset=d_query,
                        batch_size=self.eval_batch_size,
                        num_workers=self.num_workers,  # noqa
                        collate_fn=self.producer.collate_data,
                        shuffle=False,
                        drop_last=False
                    )
                else:
                    test_dataloader = DataLoader(
                        dataset=t_dataset.test,
                        batch_size=self.eval_batch_size,
                        num_workers=self.num_workers,  # noqa
                        collate_fn=self.producer.collate_data,
                        shuffle=False,
                        drop_last=False
                    )
                test_metric_result, test_loss_result = self.model.evaluate(
                    dataloader=test_dataloader, support_set=support_set, s_protos_dict=self.s_protos_dict,
                    meta_train=False, valid_labels=self.valid_labels_record[f"{s_lang}_to_{t_lang}"]
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
            batch_size: int,
            anneal_mode: str,
    ):
        scheduler = self.get_scheduler(
            scheduler_name=scheduler_name, dataset_size=dataset_size, batch_size=batch_size,
            optimizer=optimizer, initial_learning_rate=initial_learning_rate,
            epochs_tot=epochs_tot, warmup_fraction=self.scheduler_cfg.warmup_fraction,
            cycle_momentum=self.scheduler_cfg.cycle_momentum, anneal_factor=self.scheduler_cfg.anneal_factor,
            anneal_mode=anneal_mode, patience=self.scheduler_cfg.patience,
            initial_extra_patience=self.scheduler_cfg.initial_extra_patience
        )
        return scheduler

    def get_valid_labels_lang_pair(self, s_lang: str, langs: List = None) -> Dict:
        valid_label_record = dict()
        if langs is None:
            langs = list(self.meta_dataset.id2lang.values())
        for lang in langs:
            valid_labels = self.select_valid_labels(
                s_lang, lang, task=self.trainer_cfg.task, threshold=self.trainer_cfg.threshold
            )
            valid_label_record[f"{s_lang}_to_{lang}"] = valid_labels
        return valid_label_record

    @staticmethod
    def select_valid_labels(s_lang: str, t_lang: str, task: str, threshold: int = 10) -> List:
        ud_stat_dir = elephant.output_root / "ud_statistics"
        with open(ud_stat_dir / f"{task}.json", mode="r") as f:
            rows = f.readlines()
            record = json.loads(rows[-1])
            stats = record["statistics"]
            s_num_samples_per_class = np.array(stats[s_lang])
            t_num_samples_per_class = np.array(stats[t_lang])

            s_labels = list(set(np.argwhere(s_num_samples_per_class >= threshold).flatten().tolist()))
            t_labels = list(set(np.argwhere(t_num_samples_per_class >= threshold).flatten().tolist()))
            valid_labels = sorted(np.intersect1d(s_labels, t_labels).tolist())

        return valid_labels

    def _build_adaptor_optimizer(self, optimizer_cfg) -> torch.optim.Optimizer:
        if optimizer_cfg.name == "Adam":
            optimizer_adaptor = torch.optim.Adam(
                self.model.proto_module.adaptor.parameters(),
                lr=optimizer_cfg.lr_adaptor,
                betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
                weight_decay=optimizer_cfg.weight_decay_adaptor
            )
        elif optimizer_cfg.name == "SGD":
            optimizer_adaptor = torch.optim.SGD(
                self.model.proto_module.adaptor.parameters(),
                lr=optimizer_cfg.lr_adaptor,
                weight_decay=optimizer_cfg.weight_decay_adaptor,
                momentum=optimizer_cfg.momentum
            )
        else:
            raise NotImplementedError(f'Optimizer "{optimizer_cfg.name}" is not implemented.')
        return optimizer_adaptor

    def _build_optimizer(self, optimizer_cfg) -> torch.optim.Optimizer:
        if optimizer_cfg.name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.proto_module.parameters(),
                lr=optimizer_cfg.lr_proto,
                betas=(optimizer_cfg.beta1, optimizer_cfg.beta2),
                weight_decay=optimizer_cfg.weight_decay_proto
            )
        elif optimizer_cfg.name == "SGD":
            optimizer = torch.optim.SGD(
                self.model.proto_module.parameters(),
                lr=optimizer_cfg.lr_proto,
                weight_decay=optimizer_cfg.weight_decay_proto,
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
            # steps_per_epoch = int(dataset_size)
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
            # steps_per_epoch = int(dataset_size)
            # num_train_steps = steps_per_epoch * self.max_epochs
            num_train_steps = steps_per_epoch * epochs_tot
            num_warmup_steps = num_train_steps * warmup_fraction if warmup_fraction > 0 else 1
            scheduler = LinearSchedulerWithWarmup(
                optimizer,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
            )
        elif scheduler_name == "AnnealOnPlateau":
            scheduler = AnnealOnPlateau(
                optimizer,
                mode=anneal_mode,
                factor=anneal_factor,
                patience=patience,
                verbose=True,
                initial_extra_patience=initial_extra_patience,
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
