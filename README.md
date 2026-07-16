# *Radiance-Field Guided Pretraining: Scaling Localization Models with Unlabeled Wireless Signals* ( ACM IMWUT/Ubicomp 2026)



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



## Data Layout

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

Dataset could be found [here]([https://wireless-spectrum.org/](https://wireless-spectrum.org/)).

## Run Training

Single GPU:

```bash
python3 pre_runner.py --config configs/pre.yaml --mode train
```

Multi-GPU with mixed precision:

Set `--num_processes` to the number of GPUs you want to use:

```bash
accelerate launch --num_processes 8 --multi_gpu --mixed_precision fp16 \
  pre_runner.py --config configs/pre.yaml --mode train
```



## License

This repository is released under `CC-BY-NC 4.0` (Attribution-NonCommercial
4.0 International). See `LICENSE`.

Parts of the model implementation are derived from Meta's MAE codebase[https://github.com/facebookresearch/mae](https://github.com/facebookresearch/mae)). In particular, `pre_model.py` retains the upstream copyright and attribution
headers. Please keep the attribution notices intact when redistributing or
modifying this repository.