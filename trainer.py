# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict
import inspect
import copy

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt

import logging
from contextlib import nullcontext
from torch.cuda.amp import GradScaler as CudaGradScaler, autocast as cuda_autocast
from torch.utils import checkpoint

try:
    from torch.amp import autocast as torch_autocast
    from torch.amp import GradScaler as TorchGradScaler
except (ImportError, AttributeError):
    torch_autocast = None
    TorchGradScaler = None
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer(object):
    def __init__(self,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 loss_config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 use_mixed_precision=False,
                 gradient_checkpointing=False,
                 checkpoint_use_reentrant=None,
                 ema_decay=0.0):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.loss_config = loss_config or {}
        self.device = device
        self.finish_train = False
        self.logger = logger
        device_type = torch.device(self.device).type if isinstance(self.device, (str, torch.device)) else "cpu"
        self.use_amp = bool(use_mixed_precision and device_type == "cuda")
        if TorchGradScaler is not None:
            scaler_kwargs = {"enabled": self.use_amp}
            try:
                signature = inspect.signature(TorchGradScaler.__init__)
            except (TypeError, ValueError):
                signature = None

            if signature is not None:
                parameters = signature.parameters
                if "device_type" in parameters:
                    scaler_kwargs["device_type"] = device_type
                elif "device" in parameters:
                    scaler_kwargs["device"] = device_type

            try:
                self.scaler = TorchGradScaler(**scaler_kwargs)
            except TypeError:
                scaler_kwargs.pop("device_type", None)
                scaler_kwargs.pop("device", None)
                self.scaler = TorchGradScaler(**scaler_kwargs)
            if self.use_amp:
                self.logger.info("Using mixed precision scaling with torch.amp.GradScaler")
        else:
            self.scaler = CudaGradScaler(enabled=self.use_amp)
            if self.use_amp:
                self.logger.info("Using mixed precision scaling with torch.cuda.amp.GradScaler")
        if self.use_amp:
            if torch_autocast is not None:
                def autocast_cm():
                    return torch_autocast(device_type=device_type)

                self.logger.info("Using mixed precision training with torch.amp.autocast")
            else:
                autocast_cm = cuda_autocast
                self.logger.info("Using mixed precision training with torch.cuda.amp.autocast")
        else:
            autocast_cm = nullcontext
        self._autocast_cm = autocast_cm
        self.gradient_checkpointing = bool(gradient_checkpointing and device_type == "cuda")
        if gradient_checkpointing and device_type != "cuda":
            self.logger.warning("Gradient checkpointing requested but CUDA is unavailable; disabling.")
        self._checkpoint_kwargs = {}
        self.gradient_checkpoint_use_reentrant = None
        self._checkpoint_supports_use_reentrant = False
        if self.gradient_checkpointing:
            self.logger.info("Gradient checkpointing enabled for training")
            try:
                checkpoint_signature = inspect.signature(checkpoint.checkpoint)
            except (TypeError, ValueError):
                checkpoint_signature = None

            if checkpoint_signature is not None and "use_reentrant" in checkpoint_signature.parameters:
                self._checkpoint_supports_use_reentrant = True

            desired_use_reentrant = checkpoint_use_reentrant
            if desired_use_reentrant is None and self._checkpoint_supports_use_reentrant:
                desired_use_reentrant = False

            if desired_use_reentrant is not None and not self._checkpoint_supports_use_reentrant:
                self.logger.warning(
                    "This PyTorch version does not support the use_reentrant flag; proceeding with the default checkpoint behaviour."
                )
                desired_use_reentrant = None

            self.gradient_checkpoint_use_reentrant = desired_use_reentrant

            if self.gradient_checkpoint_use_reentrant is not None:
                self.logger.info(
                    "Gradient checkpointing will run with use_reentrant=%s",
                    self.gradient_checkpoint_use_reentrant,
                )
                self._checkpoint_kwargs["use_reentrant"] = self.gradient_checkpoint_use_reentrant

        self.ema_decay = float(ema_decay) if ema_decay is not None else 0.0
        self.ema_model = None
        if self.ema_decay > 0.0:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            self.logger.info("Model EMA enabled with decay %.6f", self.ema_decay)

    def save_checkpoint(self, checkpoint_path, use_ema_model=False):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
            use_ema_model (bool): Whether to save EMA weights as the primary
                model state.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        model_state = self.model.state_dict()
        if self.ema_model is not None:
            ema_state = self.ema_model.state_dict()
            state_dict["ema_model"] = ema_state
            state_dict["ema_decay"] = self.ema_decay
            if use_ema_model:
                state_dict["model_non_ema"] = model_state
                model_state = ema_state
        state_dict["model"] = model_state

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False, load_ema_as_model=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
            load_ema_as_model (bool): Load EMA parameters instead of raw
                weights when available.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if load_ema_as_model and "ema_model" in state_dict:
            model_key = "ema_model"
        elif not load_ema_as_model and "model_non_ema" in state_dict:
            model_key = "model_non_ema"
        else:
            model_key = "model"
        self._load(state_dict[model_key], self.model)

        self.ema_decay = state_dict.get("ema_decay", self.ema_decay)
        ema_state = state_dict.get("ema_model")
        if self.ema_decay > 0.0:
            if self.ema_model is None:
                self.ema_model = copy.deepcopy(self.model)
                self.ema_model.to(self.device)
                for param in self.ema_model.parameters():
                    param.requires_grad_(False)
            if ema_state is not None:
                self._load(ema_state, self.ema_model)
            else:
                # Synchronize EMA weights with the loaded model when no EMA
                # snapshot is available in the checkpoint.
                self.ema_model.load_state_dict(self.model.state_dict())
                self.logger.info(
                    "EMA state missing from checkpoint; initialized EMA weights "
                    "from current model parameters."
                )

        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])

            # overwrite schedular argument parameters
            state_dict["scheduler"].update(**self.config.get("scheduler_params", {}))
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    def _update_ema(self):
        if self.ema_model is None:
            return

        with torch.no_grad():
            ema_params = list(self.ema_model.parameters())
            model_params = list(self.model.parameters())
            for ema_param, model_param in zip(ema_params, model_params):
                ema_param.mul_(self.ema_decay).add_(model_param.detach(), alpha=1.0 - self.ema_decay)

            ema_buffers = list(self.ema_model.buffers())
            model_buffers = list(self.model.buffers())
            for ema_buffer, model_buffer in zip(ema_buffers, model_buffers):
                ema_buffer.copy_(model_buffer)

    def _f0_to_loss_scale(self, f0_values):
        """Convert F0 values to the configured regression scale."""

        scale_type = str(self.loss_config.get('f0_loss_scale', 'log_hz')).lower()
        if scale_type in {'log_hz', 'log-hz', 'loghz', 'log'}:
            min_hz = float(self.loss_config.get('f0_log_min_hz', 1e-2))
            clamp_min = torch.tensor(min_hz, dtype=f0_values.dtype, device=f0_values.device)
            safe_values = torch.maximum(f0_values, clamp_min)
            return torch.log(safe_values)
        if scale_type in {'cents', 'cent'}:
            min_hz = float(self.loss_config.get('f0_log_min_hz', 1e-2))
            reference_hz = float(self.loss_config.get('f0_cents_reference_hz', 10.0))
            clamp_min = torch.tensor(min_hz, dtype=f0_values.dtype, device=f0_values.device)
            safe_values = torch.maximum(f0_values, clamp_min)
            reference = torch.tensor(reference_hz, dtype=f0_values.dtype, device=f0_values.device)
            return 1200.0 * torch.log2(safe_values / reference)
        if scale_type in {'linear_hz', 'hz', 'linear'}:
            return f0_values
        raise ValueError(f"Unsupported F0 loss scale: {scale_type}")

    def _compute_f0_loss(self, f0_pred, f0_target, silence_labels):
        """Compute the F0 regression loss masked to voiced frames only."""

        if f0_pred.dim() > f0_target.dim():
            f0_pred = f0_pred.squeeze(-1)
        voiced_mask = (silence_labels < 0.5).to(f0_pred.dtype)
        pred_in_scale = self._f0_to_loss_scale(f0_pred)
        target_in_scale = self._f0_to_loss_scale(f0_target)
        loss_per_frame = self.criterion['f0'](pred_in_scale, target_in_scale)
        masked_loss = loss_per_frame * voiced_mask
        normaliser = voiced_mask.sum().clamp_min(1.0)
        loss_f0 = masked_loss.sum() / normaliser
        lambda_f0 = float(self.loss_config.get('lambda_f0', 1.0))
        return lambda_f0 * loss_f0

    def run(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        batch = [b.to(self.device, non_blocking=True) for b in batch]

        x, f0, sil = batch
        autocast_context = self._autocast_cm

        with autocast_context():
            if self.gradient_checkpointing:
                x = x.requires_grad_()

                def forward_fn(inp):
                    return self.model(inp.transpose(-1, -2))

                f0_pred, sil_pred = checkpoint.checkpoint(forward_fn, x, **self._checkpoint_kwargs)
            else:
                f0_pred, sil_pred = self.model(x.transpose(-1, -2))

            loss_f0 = self._compute_f0_loss(f0_pred, f0, sil)
            loss_sil = self.criterion['ce'](sil_pred, sil)
            loss = loss_f0 + loss_sil

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self._update_ema()
        self.scheduler.step()

        return {'loss': loss.item(),
                'f0': loss_f0.item(),
                'sil': loss_sil.item()}

    def _train_epoch(self):
        self.epochs += 1
        train_losses = defaultdict(list)
        self.model.train()
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):
            losses = self.run(batch)
            for key, value in losses.items():
                train_losses["train/%s" % key].append(value)

        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        train_losses['train/learning_rate'] = self._get_lr()
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        eval_model = self.ema_model if self.ema_model is not None else self.model
        base_model_training = self.model.training
        eval_model_training = eval_model.training
        eval_model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device, non_blocking=True) for b in batch]
            x, f0, sil = batch

            autocast_context = self._autocast_cm
            with autocast_context():
                f0_pred, sil_pred = eval_model(x.transpose(-1, -2))

                loss_f0 = self._compute_f0_loss(f0_pred, f0, sil)
                loss_sil = self.criterion['ce'](sil_pred, sil)
                loss = loss_f0 + loss_sil


            eval_losses["eval/loss"].append(loss.item())
            eval_losses["eval/f0"].append(loss_f0.item())
            eval_losses["eval/sil"].append(loss_sil.item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        eval_model.train(eval_model_training)
        self.model.train(base_model_training)
        return eval_losses
