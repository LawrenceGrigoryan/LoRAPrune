from transformers.trainer import (
    Trainer,
    TrainerState,
    TrainOutput,
    has_length,
    is_sagemaker_mp_enabled,
    get_model_param_count,
    speed_metrics,
    deepspeed_init,
    TRAINER_STATE_NAME,
)
from transformers.trainer_callback import ExportableState
import loraprune.utils as utils
from loraprune.optimizer import LoRAPre
import math
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import is_torch_xla_available, is_apex_available
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
import os
from packaging import version
import shutil
from loguru import logger


if is_apex_available():
    logger.info("Apex is available. Using Apex for mixed precision training.")
    from apex import amp

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

is_torch_less_than_1_11 = parsed_torch_version_base < version.parse("1.11")


class LoRAPruneTrainer(Trainer):
    def __init__(self, model,
                 train_dataset,
                 eval_dataset,
                 args,
                 data_collator,
                 ratio,
                 init_ratio,
                 warmup_iters,
                 cooldown_iters,
                 prune_freq,
                 prune_metric,
                 adaptive_ema: bool = False,
                 granular_gqa: bool = False,
                 optimizer_name: str = 'adamw_torch',
                 lorapre_rank: int = 8,
                 ):
        super().__init__(model=model,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         args=args,
                         data_collator=data_collator
                         )
        self.ratio = ratio
        self.init_ratio = init_ratio
        self.warmup_iters = warmup_iters
        self.cooldown_iters = cooldown_iters
        self.prune_freq = prune_freq
        self.prune_metric = prune_metric
        self.adaptive_ema = adaptive_ema
        self.granular_gqa = granular_gqa
        self.optimizer_name = optimizer_name
        self.lorapre_rank = lorapre_rank

    def create_optimizer(self):
        """
        Build the optimizer, honouring the ``optimizer_name`` selection.

        For anything other than ``'lorapre'`` this defers to the stock
        ``Trainer.create_optimizer``, which reads ``args.optim`` as before.  For
        ``'lorapre'`` it builds ``loraprune.optimizer.LoRAPre`` over the same two
        decay/no-decay parameter groups the base class would have used, so the
        only thing that changes is the update rule.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer, also assigned to ``self.optimizer``.

        Notes
        -----
        Adam hyperparameters come from ``TrainingArguments`` (``learning_rate``,
        ``adam_beta1``, ``adam_beta2``, ``adam_epsilon``, ``weight_decay``) so
        both optimizers are configured through one set of knobs.
        """
        if self.optimizer_name != 'lorapre':
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            # Same decay/no-decay split the base Trainer applies: no weight decay
            # on biases and normalisation weights.
            decay_parameters = self.get_decay_parameter_names(opt_model)
            trainable = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in trainable if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in trainable if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = LoRAPre(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                rank=self.lorapre_rank,
            )
            self._log_lorapre_coverage([p for _, p in trainable])

        return self.optimizer

    def _log_lorapre_coverage(self, params):
        """
        Report how many trainable tensors actually take LoRA-Pre's low-rank path.

        Parameters
        ----------
        params : list of torch.Tensor
            The trainable parameters handed to the optimizer.

        Returns
        -------
        None

        Notes
        -----
        Worth logging loudly because the answer is easy to get wrong: LoRA-Pre
        only factorises a matrix when ``min(p, q) > lorapre_rank``.  With the
        default ``prune_metric='lora'`` the trainable set is the LoRA adapters,
        whose shapes are ``(lora_r, in)`` and ``(out, lora_r)``, so
        ``min(p, q) == lora_r``.  Leaving ``lorapre_rank >= lora_r`` therefore
        sends *every* tensor to the AdamW fallback and LoRA-Pre becomes a no-op.
        """
        low_rank = [p for p in params if LoRAPre.uses_low_rank(p, self.lorapre_rank)]
        fallback = [p for p in params if not LoRAPre.uses_low_rank(p, self.lorapre_rank)]

        logger.info(
            f"LoRA-Pre optimizer (rank={self.lorapre_rank}): "
            f"{len(low_rank)} tensors on the low-rank path, "
            f"{len(fallback)} on the AdamW fallback"
        )

        if not low_rank:
            logger.warning(
                f"LoRA-Pre is a no-op: no trainable tensor has min(shape) > "
                f"lorapre_rank={self.lorapre_rank}, so every one falls back to "
                f"AdamW. With prune_metric='lora' the trainable tensors are the "
                f"LoRA adapters, so lorapre_rank must be strictly below lora_r."
            )
            return

        # Adam keeps two full moments per tensor; LoRA-Pre keeps four thin factors.
        adam_elems = sum(2 * p.numel() for p in low_rank)
        lorapre_elems = sum(
            2 * (p.shape[0] + p.shape[1]) * self.lorapre_rank for p in low_rank
        )
        logger.info(
            f"LoRA-Pre optimizer state on the low-rank path: {lorapre_elems:,} vs "
            f"{adam_elems:,} elements for Adam ({lorapre_elems / adam_elems:.3f}x)"
        )

    def _log_cuda_memory(self, tag):
        """
        Log CUDA memory usage.

        Parameters
        ----------
        tag : str
            Short label identifying where in training this reading was taken.

        Returns
        -------
        None

        Notes
        -----
        Peak allocation is the number that decides whether a run fits on the
        card.  It moves less than the optimizer state alone does: LoRA-Pre
        reconstructs ``m_B @ m_A`` into a transient ``p x q`` buffer each step,
        giving back at peak some of what it saves in state.
        """
        if not torch.cuda.is_available():
            return
        mib = 1024**2
        logger.info(
            f"[{tag}] CUDA memory: "
            f"allocated={torch.cuda.memory_allocated() / mib:.0f} MiB, "
            f"peak={torch.cuda.max_memory_allocated() / mib:.0f} MiB, "
            f"reserved={torch.cuda.memory_reserved() / mib:.0f} MiB"
        )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        if args.deepspeed:
            logger.info("Deepspeed is enabled, initializing deepspeed engine and getting the wrapped model.")
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
        total_params = kept_params = sum([p.numel() if not p.requires_grad else 0 for p in model.parameters()])
        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")


        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)

        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        if self.prune_metric == 'grad':
            utils.unfreeze(model)

        # Reset the peak counter so it measures the training loop rather than
        # model loading; otherwise the optimizers are not comparable.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._log_cuda_memory("before training")

        sensitivity_dict = utils.init_sensitivity_dict(model, granular_gqa=self.granular_gqa)
        if self.adaptive_ema:
            var_dict, alpha_dict, count_dict = utils.init_adaptive_ema_state(model, granular_gqa=self.granular_gqa)
        else:
            var_dict = alpha_dict = count_dict = None
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)


            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer), args.max_grad_norm
                            )
                        else:
                            # accelerator.clip_grad_norm_ unscales AMP-scaled gradients before
                            # clipping, so the reported norm and the actual clip are both in
                            # true (unscaled) gradient space.  The old nn.utils.clip_grad_norm_
                            # path clipped scaled gradients, making max_grad_norm effectively ~0.
                            grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(), args.max_grad_norm
                            )
                    else:
                        grad_norm = None

                    # Optimizer step
                    if not self.deepspeed:
                        sensitivity_dict = utils.update_sensitivity_dict(
                            model, sensitivity_dict, self.prune_metric,
                            adaptive_ema=self.adaptive_ema,
                            var_dict=var_dict,
                            alpha_dict=alpha_dict,
                            count_dict=count_dict,
                            granular_gqa=self.granular_gqa,
                        )
                    ratio = utils.schedule_sparsity_ratio(self.state.global_step, self.state.max_steps,
                                                          self.warmup_iters,
                                                          self.cooldown_iters, self.init_ratio, self.ratio)

                    # ratio = 0.05
                    if (self.state.global_step) % self.prune_freq == 0 and ratio > self.init_ratio and ratio < self.ratio:
                        utils.local_prune(model, sensitivity_dict, ratio, self.ratio, granular_gqa=self.granular_gqa)

                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    # Optimizer state is allocated lazily on the first step, so
                    # this is the earliest reading that includes it.
                    if self.state.global_step == 0:
                        self._log_cuda_memory("after first step")

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    _gs = self.state.global_step
                    if _gs % args.logging_steps == 0:
                        _gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm or 0.0)
                        _lr = self.optimizer.param_groups[0]['lr']
                        logger.info(f"[step={_gs}/{self.state.max_steps}], training loss={tr_loss_step.item():.4f}, grad_norm={_gn:.4f}, lr={_lr:.2e}")
                        self._log_cuda_memory(f"step={_gs}")
                    
                    self._maybe_log_save_evaluate(tr_loss, grad_norm if grad_norm is not None else None, model, trial, epoch, ignore_keys_for_eval, start_time)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, None, model, trial, epoch, ignore_keys_for_eval, start_time)


            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        self._log_cuda_memory("end of training")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)