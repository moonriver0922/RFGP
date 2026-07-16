#!/usr/bin/env python3
"""Minimal end-to-end smoke test for CI (CPU or GPU).

Creates tiny synthetic BLE tensors, runs 1 training epoch, and exits 0 on success.
"""
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import resolve_paths
from pre_runner import PretrainRunner

FEATURE_DIM = 332
CHUNKSIZE = 64
N_SAMPLES = 128
SCENE_ID = 7


def _make_shard(n_samples: int, scene_id: int) -> torch.Tensor:
    feat = torch.zeros(n_samples, FEATURE_DIM)
    feat[:, 0] = scene_id
    feat[:, 1:4] = torch.tensor([0.0, 0.0, 1.0])  # gateway xyz
    feat[:, 4] = 1.0  # quaternion w
    feat[:, 8:] = torch.rand(n_samples, 324)
    return feat


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for split in ('train', 'test'):
            d = root / 'data' / 'ble' / split
            d.mkdir(parents=True)
            torch.save(_make_shard(N_SAMPLES, SCENE_ID), d / f'{split}_{SCENE_ID}.t')

        with open(ROOT / 'configs' / 'pre.yaml') as f:
            cfg = yaml.safe_load(f)

        cfg['path']['train_data_roots'] = {'ble': str(root / 'data/ble/train')}
        cfg['path']['test_data_roots'] = {'ble': str(root / 'data/ble/test')}
        cfg['path']['expname'] = 'ci/'
        cfg['training']['total_epochs'] = 1
        cfg['training']['warmup_epochs'] = 0
        cfg['training']['data_fraction'] = 1.0
        cfg['training']['resample_every_epochs'] = 0

        cfg['path'] = resolve_paths(cfg['path'], str(ROOT / 'configs/pre.yaml'))
        cfg['path']['logdir'] = str(root / 'outputs/logs/pretrain')
        cfg['path']['train_fig_path'] = str(root / 'outputs/figs/pretrain')
        cfg['path']['test_fig_path'] = str(root / 'outputs/figs/pretrain')
        cfg['path']['test_pos_path'] = str(root / 'outputs/pos/pretrain')

        # CI runners have no GPU; training still exercises the full forward path on CPU.
        if not torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ''

        worker = PretrainRunner(
            mode='train',
            config_path=str(ROOT / 'configs/pre.yaml'),
            **cfg,
        )
        worker.train_network()
        print('CI smoke passed')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
