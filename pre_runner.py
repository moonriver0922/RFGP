# -*- coding: utf-8 -*-
"""RFGP spectrum pretraining runner.
"""
import argparse
import os
from shutil import copyfile

import accelerate
import matplotlib.image as plm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from accelerate.utils import DistributedDataParallelKwargs
from pytorch_msssim import SSIM
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset
from tqdm import tqdm

from dataset import MyDataset, discover_scene_ids
from logger import logger_config
from paths import default_config_path, resolve_paths
from pre_model import RFGP

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

dis2mse = lambda x, y: torch.mean((x - y) ** 2)
ssim_module = SSIM(data_range=1, size_average=True, win_size=3, channel=1)

torch.manual_seed(3402)
np.random.seed(3402)


class PretrainRunner:
    """Accelerate-based trainer for RFGP spectrum reconstruction pretraining.
    """

    def __init__(self, mode, config_path=None, **kwargs) -> None:
        """
        Parameters
        ----------
        mode : str
            ``'train'`` or ``'test'``.
        config_path : str, optional
            Path to the YAML config (used to resolve relative paths).
        **kwargs
            Parsed YAML sections: path / dataset / networks / training.
        """
        kwargs_path = resolve_paths(kwargs['path'], config_path)
        kwargs['path'] = kwargs_path
        kwargs_dataset = kwargs['dataset']
        kwargs_network = kwargs['networks']
        kwargs_train = kwargs['training']

        self.expname = kwargs_path['expname']
        self.logdir = kwargs_path['logdir']
        self.load_ckpt = kwargs_path['load_ckpt']
        self.devices = torch.device('cuda')
        self.test_pos_path = kwargs_path['test_pos_path']
        self.pre_ckpts = kwargs_path.get('pre_ckpts') or ''
        self.test_pos_path = os.path.join(self.test_pos_path, self.expname)
        os.makedirs(self.test_pos_path, exist_ok=True)

        self.train_fig_path = os.path.join(
            kwargs_path['train_fig_path'], f'{self.expname}train/')
        self.test_fig_path = os.path.join(
            kwargs_path['test_fig_path'], f'{self.expname}test/')
        os.makedirs(self.train_fig_path, exist_ok=True)
        os.makedirs(self.test_fig_path, exist_ok=True)

        self.n_seq = kwargs_dataset['n_seq']
        self.batch_size = kwargs_train['batch_size']
        self.chunksize = kwargs_train['chunksize']
        self.total_epochs = kwargs_train['total_epochs']
        self.loss = kwargs_train['loss']
        self.beta = kwargs_train['beta']
        self.i_save = kwargs_train['i_save']
        self.loc_type = kwargs_train['loc_type']
        self.base_lr = kwargs_train['base_lr']
        self.final_lr = kwargs_train['final_lr']
        self.warmup_epochs = kwargs_train['warmup_epochs']
        self.start_warmup_lr = kwargs_train['start_warmup_lr']
        # Subset training: keep full pool on CPU, put a fraction on the device.
        # Defaults match configs/pre.yaml: 30% per scene, redraw every 20 epochs.
        self.data_fraction = float(kwargs_train.get('data_fraction', 0.3))
        self.resample_every_epochs = int(kwargs_train.get('resample_every_epochs', 20))
        self.mask_id = 0
        self.error_median_over_epoch = []
        self.error_std_over_epoch = []
        self._subset_rng = torch.Generator()
        self._subset_rng.manual_seed(3402)
        self._subset_round = 0

        log_savepath = os.path.join(self.logdir, self.expname, 'logger.log')
        os.makedirs(os.path.dirname(log_savepath), exist_ok=True)
        self.logger = logger_config(log_savepath=log_savepath, logging_name='mae2')

        # --- Load data first so scene ids drive NeRF construction ---
        train_roots = kwargs_path.get('train_data_roots') or {}
        test_roots = kwargs_path.get('test_data_roots') or {}
        train_set = MyDataset(train_roots, self.chunksize)
        try:
            test_set = MyDataset(test_roots, self.chunksize) if test_roots else None
        except FileNotFoundError:
            test_set = None
            self.logger.warning(
                "No test tensors under %s; pred() will be skipped.", test_roots)

        self.scene_ids = discover_scene_ids(train_set, test_set)
        self.train_set = train_set
        for line in train_set.modality_report('train').splitlines():
            self.logger.info(line)
        if test_set is not None:
            for line in test_set.modality_report('test').splitlines():
                self.logger.info(line)
        else:
            self.logger.info("[test] skipped (no usable test tensors)")

        self.logger.info(
            "train subsetting: data_fraction=%s, resample_every_epochs=%s",
            self.data_fraction, self.resample_every_epochs)

        if test_set is not None:
            (test_enc_token, self.test_spt, test_dec_token, self.test_label,
             self.test_ts, self.test_id, self.test_gateways) = test_set.loaddata()
            test_dataset = TensorDataset(test_enc_token, test_dec_token)
            self.test_iter = torch.utils.data.DataLoader(
                test_dataset, self.batch_size, pin_memory=True, shuffle=False, drop_last=True)
        else:
            self.test_spt = self.test_label = None
            self.test_ts = self.test_id = self.test_gateways = None
            self.test_iter = None

        # Placeholder; real train loader is built after Accelerator exists.
        self.train_iter = None
        self.transform_iter = None
        self.train_spt = self.train_label = None
        self.train_ts = self.train_id = self.train_gateways = None

        self.accelerator = accelerate.Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        self._install_train_subset(seed=3402, prepare_loader=False)

        self.mae = RFGP(scene_ids=self.scene_ids, **kwargs_network).to(
            self.accelerator.device)
        self.optimizer = optim.Adam(
            params=list(self.mae.parameters()),
            lr=float(kwargs_train['base_lr']),
            weight_decay=float(kwargs_train['weight_decay']))

        self.epoch_start = 1
        self.step = 0
        if kwargs_path.get('load_pretrain') and mode == 'train' and self.pre_ckpts:
            self.load_pretrained()
        if kwargs_path.get('load_ckpt') or mode == 'test':
            self.load_checkpoints()
        self.current_epoch = self.epoch_start

        if self.accelerator.is_local_main_process:
            self.logger.info("expname:%s, logdir:%s", self.expname, self.logdir)
        self.logger_tb = SummaryWriter(os.path.join(self.logdir, self.expname, 'tensorboard'))

        total_params = sum(p.numel() for p in self.mae.parameters())
        if self.accelerator.is_local_main_process:
            self.logger.info('Total parameters: %s', total_params)
            self.logger.info(
                "NeRF bank: %s scene(s) %s",
                len(self.scene_ids), self.scene_ids)
            far_preview = {
                sid: self.mae.renderers[str(sid)].far
                for sid in self.scene_ids[:min(8, len(self.scene_ids))]
            }
            self.logger.info("renderer far preview: %s", far_preview)
            self.logger.debug(
                "train_iter length:%s, test_iter length:%s",
                len(self.train_iter.dataset),
                len(self.test_iter.dataset) if self.test_iter is not None else 0)
        self.train_iter, self.mae, self.optimizer = self.accelerator.prepare(
            self.train_iter, self.mae, self.optimizer)

    def _install_train_subset(self, seed: int, prepare_loader: bool = True):
        """Sample a per-scene training subset from the CPU pool and rebuild the loader.

        Parameters
        ----------
        seed : int
            RNG seed for this subset draw.
        prepare_loader : bool
            If True, wrap the new DataLoader with ``accelerator.prepare``.
            Set False during ``__init__`` before the joint prepare of model/optim.
        """
        self._subset_rng.manual_seed(int(seed))
        indices = self.train_set.sample_chunk_indices(
            self.data_fraction, generator=self._subset_rng)
        (train_enc_token, train_spt, train_dec_token, train_label,
         train_ts, train_id, train_gateways) = self.train_set.gather(indices)

        # Keep active training tensors on CPU here; train_network moves them once
        self._train_spt_cpu = train_spt
        self._train_label_cpu = train_label
        self._train_ts_cpu = train_ts
        self._train_id_cpu = train_id
        self._train_gateways_cpu = train_gateways
        self._active_indices = indices
        self._subset_round += 1

        train_dataset = TensorDataset(train_enc_token, train_dec_token)
        self.transform_iter = torch.utils.data.DataLoader(
            train_dataset, self.batch_size, pin_memory=True, shuffle=False, drop_last=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, self.batch_size, pin_memory=True, shuffle=True, drop_last=True)

        if prepare_loader:
            self.train_iter = self.accelerator.prepare(train_loader)
        else:
            self.train_iter = train_loader

        for line in self.train_set.subset_report(
                indices, self.data_fraction, split='train').splitlines():
            self.logger.info(line)
        self.logger.info(
            "train subset round=%s seed=%s active_chunks=%s",
            self._subset_round, seed, indices.numel())

    def _materialize_train_tensors(self):
        """Copy the active CPU subset onto the accelerator device."""
        device = self.accelerator.device
        self.train_spt = self._train_spt_cpu.to(device, non_blocking=True)
        self.train_label = self._train_label_cpu.to(device, non_blocking=True)
        self.train_ts = self._train_ts_cpu.to(device, non_blocking=True)
        self.train_id = self._train_id_cpu.to(device, non_blocking=True)
        self.train_gateways = self._train_gateways_cpu.to(device, non_blocking=True)

    def load_pretrained(self):
        """Load a pretrained checkpoint into the model."""
        pretrain_dir = self.pre_ckpts
        self.logger.info("Load pretrained model from %s", pretrain_dir)
        ckpt = torch.load(pretrain_dir, map_location=self.accelerator.device)
        pretrained_dict = ckpt['mae_state_dict']
        model_dict = self.mae.state_dict()

        for k in model_dict.keys():
            if 'mae1.' in k:
                unwarp_k = k.replace("mae1.", "")
            elif 'mae2.' in k:
                unwarp_k = k.replace("mae2.", "")
            elif 'mae3.' in k:
                unwarp_k = k.replace("mae3.", "")
            else:
                unwarp_k = k
            if unwarp_k in pretrained_dict.keys():
                if model_dict[k].shape == pretrained_dict[unwarp_k].shape:
                    model_dict[k] = pretrained_dict[unwarp_k]

        self.mae.load_state_dict(model_dict)

    def load_checkpoints(self):
        """Resume from the latest checkpoint under ``logdir/expname/ckpts``."""
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        os.makedirs(ckptsdir, exist_ok=True)
        ckpts = [
            os.path.join(ckptsdir, f)
            for f in sorted(os.listdir(ckptsdir)) if 'tar' in f
        ]
        self.logger.info('Found ckpts: %s', ckpts)
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            self.logger.info('Reload from %s', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self.accelerator.device)

            new_state_dict = {}
            for k in self.mae.state_dict().keys():
                if (k in ckpt['mae_state_dict']
                        and self.mae.state_dict()[k].shape == ckpt['mae_state_dict'][k].shape):
                    new_state_dict[k] = ckpt['mae_state_dict'][k]
                else:
                    new_state_dict[k] = self.mae.state_dict()[k]

            self.mae.load_state_dict(new_state_dict)
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.epoch_start = ckpt['epoch_start'] + 1

    def saveckpts(self, step):
        """Save model/optimizer state; keep only the newest when >1 exists."""
        ckptsdir = os.path.join(self.logdir, self.expname, 'ckpts')
        os.makedirs(ckptsdir, exist_ok=True)
        model_lst = [x for x in sorted(os.listdir(ckptsdir)) if x.endswith('.tar')]
        if len(model_lst) > 1:
            os.remove(os.path.join(ckptsdir, model_lst[0]))

        ckptname = os.path.join(ckptsdir, f"{self.current_epoch:04d}{step:06d}.tar")
        unwrap_model = self.accelerator.unwrap_model(self.mae)
        unwrap_optim = self.accelerator.unwrap_model(self.optimizer)
        state_dict = {
            'epoch_start': self.current_epoch,
            'scene_ids': self.scene_ids,
            'mae_state_dict': unwrap_model.state_dict(),
            'optimizer_state_dict': unwrap_optim.state_dict(),
        }
        torch.save(state_dict, ckptname)
        if self.accelerator.is_local_main_process:
            self.logger.debug('Saved checkpoints at: %s', ckptname)

    def get_random_mask(self, B, n_seq):
        """Return an all-False mask placeholder ``[B, n_seq]``."""
        mask = torch.zeros((B, n_seq))
        mask = mask.eq(1).to(self.accelerator.device)
        return mask

    def cosine_scheduler(self, base_value, final_value, epochs, niter_per_ep,
                         warmup_epochs=0, start_warmup_value=0):
        """Cosine LR schedule with optional linear warmup."""
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        base_value = float(base_value)
        final_value = float(final_value)
        start_warmup_value = float(start_warmup_value)
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * iters / len(iters)))
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule

    def train_network(self, spt_nums=36 * 9):
        """Run spectrum reconstruction pretraining.

        Loss: ``100 * MSE + (1 - SSIM) + 0.01 * MoE_aux``.
        Checkpoints and test prediction metrics are saved every 20 epochs.

        When ``data_fraction < 1``, only a per-scene subset is trained on; if
        ``resample_every_epochs > 0``, that subset is redrawn periodically.
        """
        self.mae.train()
        self._materialize_train_tensors()
        num_batches = len(self.train_iter)
        log_step_interval = 50
        global_iter_num = 0
        self.logger.info(
            'train samples: %s, batches: %s (fraction=%s)',
            len(self.train_iter.dataset), num_batches, self.data_fraction)
        self.cosine_schedule = self.cosine_scheduler(
            self.base_lr, self.final_lr, self.total_epochs, num_batches,
            self.warmup_epochs, self.start_warmup_lr)

        train_spt = self.train_spt
        train_id = self.train_id
        train_gateways = self.train_gateways
        train_label = self.train_label

        for epoch in range(self.epoch_start, self.total_epochs + 1):
            if (self.resample_every_epochs > 0
                    and self.data_fraction < 1.0
                    and epoch > self.epoch_start
                    and ((epoch - self.epoch_start) % self.resample_every_epochs) == 0):
                self.logger.info(
                    "Resampling train subset at epoch %s (every %s epochs)",
                    epoch, self.resample_every_epochs)
                self._install_train_subset(seed=3402 + epoch, prepare_loader=True)
                self._materialize_train_tensors()
                train_spt = self.train_spt
                train_id = self.train_id
                train_gateways = self.train_gateways
                train_label = self.train_label
                num_batches = len(self.train_iter)
                remaining = self.total_epochs - epoch + 1
                self.cosine_schedule = self.cosine_scheduler(
                    self.base_lr, self.final_lr, max(remaining, 1), num_batches,
                    0, self.base_lr)
                global_iter_num = 0

            with tqdm(total=num_batches, desc=f"Epoch {epoch}/{self.total_epochs}") as pbar:
                for step, (enc_token, dec_token) in enumerate(self.train_iter):
                    B, n_seq = enc_token.shape
                    sched_idx = min(global_iter_num, len(self.cosine_schedule) - 1)
                    current_lr = self.cosine_schedule[sched_idx]
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr

                    spt = train_spt[enc_token.view(-1)].reshape(
                        B * self.chunksize, n_seq, spt_nums)
                    train_id_ = train_id[enc_token.view(-1)].reshape(
                        B * self.chunksize, n_seq, 1)
                    _id = torch.mean(train_id_).int()
                    assert torch.all(_id == train_id_), "invalid batch!"
                    gateways_info = train_gateways[enc_token.view(-1)].reshape(
                        B * self.chunksize, n_seq, -1)
                    label = train_label[dec_token.view(-1)].reshape(
                        B * self.chunksize, n_seq, spt_nums)

                    output_signals, _, masked_imgs, load_balancing_loss, count = self.mae(
                        spt, _id, gateways_info, 1)

                    local_label = label.reshape(-1, 1, 9, 36)
                    ssim_loss = 1 - ssim_module(output_signals, local_label)
                    mse_loss = dis2mse(output_signals, local_label)

                    self.optimizer.zero_grad()
                    loss_all = 100 * mse_loss + ssim_loss + 0.01 * load_balancing_loss
                    self.accelerator.backward(loss_all)
                    self.optimizer.step()

                    loss = loss_all
                    global_iter_num = global_iter_num + 1

                    if self.accelerator.is_local_main_process:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"mse_loss:{mse_loss.item():.6f}, "
                            f"ssim_loss:{ssim_loss.item():.4f}, "
                            f"load_balancing_loss:{load_balancing_loss.item():.4f}, "
                            f"id:{_id.item()}")
                        if global_iter_num % log_step_interval == 0:
                            self.logger_tb.add_scalar("loss", loss.item(), global_step=global_iter_num)
                            self.logger_tb.add_scalar(
                                "mse_loss", mse_loss.item(), global_step=global_iter_num)
                            self.logger_tb.add_scalar(
                                "ssim_loss", ssim_loss.item(), global_step=global_iter_num)

                    if (step % 500 == 0) and self.accelerator.is_local_main_process:
                        self.logger.debug("count:%s", count)
                        fig_pred = output_signals[0, 0].reshape(9, 36).detach().cpu().numpy()
                        fig_gt = local_label[0, 0].reshape(9, 36).detach().cpu().numpy()
                        fig_mask = masked_imgs[0, 0].reshape(9, 36).detach().cpu().numpy()
                        fig_all = np.concatenate([fig_gt, fig_mask, fig_pred], axis=1)
                        plm.imsave(
                            os.path.join(self.train_fig_path, f"train{epoch}_{step}.png"),
                            fig_all)

            self.current_epoch = epoch
            if (self.current_epoch % 20 == 0) and self.accelerator.is_local_main_process:
                self.saveckpts(step)
                if self.test_iter is not None:
                    pred_mse_loss, pred_ssim_loss, pred_load_loss = self.pred(
                        self.test_iter, self.test_spt, self.test_label, epoch, step)
                    self.logger.info(
                        "pred_loss:%s, %s, %s", pred_mse_loss, pred_ssim_loss, pred_load_loss)
                    self.logger_tb.add_scalar(
                        "pred_mse_loss", pred_mse_loss.item(), global_step=global_iter_num)
                    self.logger_tb.add_scalar(
                        "pred_ssim_loss", pred_ssim_loss.item(), global_step=global_iter_num)
                    self.logger_tb.add_scalar(
                        "pred_load_loss", pred_load_loss.item(), global_step=global_iter_num)

    def pred(self, dataset, spt_set, label_set, train_epoch=None, train_step=None,
             spt_nums=36 * 9):
        """Evaluate spectrum reconstruction and dump position features.

        Returns
        -------
        mse_loss, ssim_loss, load_balancing_loss : torch.Tensor
            Averaged over dataloader steps.
        """
        self.mae.eval()
        mse_loss = 0
        ssim_loss = 0
        load_balancing_loss = 0
        pos_features = []
        steps = 0

        spt_set = spt_set.to(self.accelerator.device, non_blocking=True)
        label_set = label_set.to(self.accelerator.device, non_blocking=True)
        test_id = self.test_id.to(self.accelerator.device, non_blocking=True)
        test_gateways = self.test_gateways.to(self.accelerator.device, non_blocking=True)
        test_ts = self.test_ts.to(self.accelerator.device, non_blocking=True)

        with torch.no_grad():
            for step, (enc_token, dec_token) in enumerate(dataset):
                B, n_seq = enc_token.shape
                steps += 1
                enc_token_flat = enc_token.view(-1)
                dec_token_flat = dec_token.view(-1)

                spt = spt_set[enc_token_flat].reshape(B * self.chunksize, n_seq, spt_nums)
                gateways_info = test_gateways[enc_token_flat].reshape(
                    B * self.chunksize, n_seq, -1)
                label = label_set[dec_token_flat].reshape(B * self.chunksize, n_seq, spt_nums)
                test_id_ = test_id[enc_token_flat].reshape(B * self.chunksize, n_seq, 1)
                _id = torch.mean(test_id_).int()
                assert torch.all(_id == test_id_), f"Invalid batch ID at step {step}"

                output_signals, pos_feature, masked_imgs, _load_balancing_loss, count = self.mae(
                    spt, _id, gateways_info, 1)

                local_label = label.reshape(-1, 1, 9, 36)
                _ssim_loss = 1 - ssim_module(output_signals, local_label)
                _mse_loss = dis2mse(output_signals, local_label)
                mse_loss += _mse_loss
                ssim_loss += _ssim_loss
                load_balancing_loss += _load_balancing_loss
                pos_features.append(pos_feature)

                if step % 100 == 0 and self.accelerator.is_local_main_process:
                    fig_pred = output_signals[0, 0].reshape(9, 36).cpu().numpy()
                    fig_mask = masked_imgs[0, 0].reshape(9, 36).cpu().numpy()
                    fig_gt = local_label[0, 0].reshape(9, 36).cpu().numpy()
                    fig_all = np.concatenate([fig_gt, fig_mask, fig_pred], axis=1)
                    plm.imsave(
                        os.path.join(
                            self.test_fig_path,
                            f"test{train_epoch}_{train_step}_{step}.png"),
                        fig_all)

        pos_features = torch.cat(pos_features, dim=0)
        torch.save(
            pos_features,
            os.path.join(self.test_pos_path, f"pos_features_{train_epoch}.t"))
        return mse_loss / steps, ssim_loss / steps, load_balancing_loss / steps

# Backward-compatible alias
mae_Runner = PretrainRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RFGP')
    parser.add_argument(
        '--config', type=str, default=default_config_path(),
        help='config file path (default: configs/pre.yaml next to this package)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    with open(args.config) as f:
        kwargs = yaml.safe_load(f)

    kwargs['path'] = resolve_paths(kwargs['path'], args.config)

    if args.mode == 'train':
        logdir = os.path.join(kwargs['path']['logdir'], kwargs['path']['expname'])
        os.makedirs(logdir, exist_ok=True)
        copyfile(args.config, os.path.join(logdir, 'config.yaml'))

    worker = PretrainRunner(mode=args.mode, config_path=args.config, **kwargs)
    if args.mode == 'train':
        worker.train_network()
    elif args.mode == 'test':
        pass
