# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict
import inspect

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

from hybrid_features import fuse_f0_predictions

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
                 fusion_config=None):

        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.loss_config = loss_config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.dsp_feature_names = []
        if train_dataloader is not None:
            self.dsp_feature_names = getattr(train_dataloader.dataset, "dsp_feature_names", [])
        if not self.dsp_feature_names and val_dataloader is not None:
            self.dsp_feature_names = getattr(val_dataloader.dataset, "dsp_feature_names", [])
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

        fusion_config = dict(fusion_config or config.get('fusion', {}))
        self.fusion_enabled = bool(fusion_config.get('enabled', False))
        self.fusion_neural_weight = float(fusion_config.get('neural_weight', 1.5))
        self.fusion_octave_tolerance = float(fusion_config.get('octave_tolerance_cents', 120.0))
        self.fusion_weights = fusion_config.get('weights', {}) or {}
        if self.fusion_enabled and not self.dsp_feature_names:
            self.logger.warning("Fusion requested but no DSP features are available; disabling fusion.")
            self.fusion_enabled = False

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = self.model.state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._load(state_dict["model"], self.model)

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

    def run(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        batch = [b.to(self.device, non_blocking=True) for b in batch]

        if len(batch) == 4:
            x, f0, sil, dsp = batch
        else:
            x, f0, sil = batch
            dsp = None
        autocast_context = self._autocast_cm

        with autocast_context():
            if self.gradient_checkpointing:
                x = x.requires_grad_()

                def forward_fn(inp):
                    return self.model(inp.transpose(-1, -2))

                f0_pred, sil_pred = checkpoint.checkpoint(forward_fn, x, **self._checkpoint_kwargs)
            else:
                f0_pred, sil_pred = self.model(x.transpose(-1, -2))

            loss_f0 = self.loss_config['lambda_f0'] * self.criterion['l1'](f0_pred.squeeze(), f0)
            loss_sil = self.criterion['ce'](sil_pred, sil)
            loss = loss_f0 + loss_sil

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        results = {'loss': loss.item(),
                   'f0': loss_f0.item(),
                   'sil': loss_sil.item()}
        if self.fusion_enabled and dsp is not None:
            fused = fuse_f0_predictions(
                f0_pred.squeeze(),
                dsp,
                feature_names=self.dsp_feature_names,
                weights=self.fusion_weights,
                neural_weight=self.fusion_neural_weight,
                octave_tolerance_cents=self.fusion_octave_tolerance,
            )
            fusion_loss = self.loss_config['lambda_f0'] * self.criterion['l1'](fused, f0)
            results['fused_f0'] = fusion_loss.item()
        return results

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
        self.model.eval()
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):
            batch = [b.to(self.device, non_blocking=True) for b in batch]
            if len(batch) == 4:
                x, f0, sil, dsp = batch
            else:
                x, f0, sil = batch
                dsp = None

            autocast_context = self._autocast_cm
            with autocast_context():
                f0_pred, sil_pred = self.model(x.transpose(-1, -2))

                loss_f0 = self.loss_config['lambda_f0'] * self.criterion['l1'](f0_pred.squeeze(), f0)
                loss_sil = self.criterion['ce'](sil_pred, sil)
                loss = loss_f0 + loss_sil


            eval_losses["eval/loss"].append(loss.item())
            eval_losses["eval/f0"].append(loss_f0.item())
            eval_losses["eval/sil"].append(loss_sil.item())
            if self.fusion_enabled and dsp is not None:
                fused = fuse_f0_predictions(
                    f0_pred.squeeze(),
                    dsp,
                    feature_names=self.dsp_feature_names,
                    weights=self.fusion_weights,
                    neural_weight=self.fusion_neural_weight,
                    octave_tolerance_cents=self.fusion_octave_tolerance,
                )
                fusion_loss = self.loss_config['lambda_f0'] * self.criterion['l1'](fused, f0)
                eval_losses["eval/fused_f0"].append(fusion_loss.item())

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses
