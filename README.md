# RFGP （ACM IMWUT/Ubicomp 2026）

RFGP is a spectrum pretraining pipeline for RF localization. It trains a MAE+MoE encoder together with per-scene NeRF renderers to reconstruct spatial spectrum maps.

## What Is Included

- `pre_runner.py`: train entrypoint
- `pre_model.py`: MAE + MoE + scene-aware rendering model
- `dataset.py`: tensor loader, chunking, scene discovery, subsetting
- `configs/pre.yaml`: default training config
- `requirements.txt`: Python dependencies

## Environment

Recommended:

- Python 3.8+
- PyTorch with CUDA support

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you already manage environments with Conda:

```bash
conda create -n rfgp python=3.8 -y
conda activate rfgp
pip install -r requirements.txt
```



## Data Format

Each `.t` file is loaded with `torch.load(...)`.

Supported sample layouts:

1. Real / preferred format:

```text
(N, F), F = 332
scenario(1) | gateways(7) | spectrum(324)
```

Notes:

- `scenario` is the scene id used to build the NeRF^2 dynamically.
- `gateways(7)` means `xyz(3) + quaternion(4)`.
- `spectrum(324)` is a flattened `9 x 36` spectrum map.



## Data Directory Layout

The default config expects:

```text
data/
  ble/
    train/
    test/
  wifi/
    train/
    test/
  iiot/
    train/
    test/
  rfid/
    train/
    test/
```

Put all `*.t` files for each modality directly into the corresponding
train/test directory.

Example:

```text
data/ble/train/train_0.t
data/ble/train/train_1.t
data/wifi/train/train_100.t
...
```



## Run Training

Single GPU:

```bash
python3 pre_runner.py --config configs/pre.yaml --mode train
```

Multi-GPU with mixed precision:

Set `--num_processes` to the number of GPUs you want to use:

```bash
accelerate launch --num_processes <NUM_GPUS> --multi_gpu --mixed_precision fp16 \
  pre_runner.py --config configs/pre.yaml --mode train
```



## License

This repository is released under `CC-BY-NC 4.0` (Attribution-NonCommercial
4.0 International). See `LICENSE`.

Parts of the model implementation are derived from Meta's MAE codebase. In
particular, `pre_model.py` retains the upstream copyright and attribution
headers. Please keep the attribution notices intact when redistributing or
modifying this repository.