# Distributive-ML

Distributive-ML is a collection of tools, experiments and utilities for building and evaluating distributed machine learning systems. It provides example training scripts, configuration-driven experiments, data preparation utilities, and integrations with common distributed training backends (PyTorch DDP, Horovod, torchrun). The project is intended as a starting point for research and engineering work on scaling ML training across multiple GPUs and nodes.

## Highlights

- Config-driven experiments (YAML/JSON)
- Example training loops, evaluation scripts and model definitions
- Support for single-node and multi-node distributed training (torch.distributed / Horovod)
- Utilities for data preparation, checkpointing, and logging
- Templates to help run reproducible experiments

## Repository layout

- configs/           - Example experiment configuration files
- src/               - Core library code (models, datasets, training loops)
- scripts/           - Entrypoint scripts for training, evaluation and data preparation
- data/              - Dataset downloaders and preprocessing scripts
- experiments/       - Example experiments and results
- checkpoints/       - Saved models and checkpoints (gitignored)
- docs/              - Documentation and how-tos

Note: If your repository structure differs, adapt the sections above to match actual directories and scripts.

## Requirements

- Python 3.8+
- PyTorch (recommended) or another supported backend
- CUDA toolkit for GPU training (optional)
- Optional: Horovod for MPI-based training

Install base dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use CUDA, install a matching PyTorch build, for example:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
```

## Quickstart

1. Clone the repo:

```bash
git clone https://github.com/pauldebojyoti/Distributive-ML.git
cd Distributive-ML
```

2. Prepare data (example):

```bash
python scripts/download_data.py --dataset cifar10 --out data/
python scripts/preprocess.py --input data/cifar10 --output data/processed
```

3. Single-process training (development):

```bash
python scripts/train.py --config configs/resnet_cifar10.yaml --device cuda
```

4. Multi-GPU on a single node (torchrun / torch.distributed):

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/resnet_cifar10.yaml
```

5. Multi-node using Horovod (example):

```bash
horovodrun -np 8 -H host1:4,host2:4 python scripts/train_horovod.py --config configs/resnet_cifar10.yaml
```

Adjust the example commands above to match the actual script names and CLI arguments in this repository.

## Configuration

Experiments are driven by configuration files under `configs/`. A typical config includes:

- model: architecture and hyperparameters
- dataset: name and preprocessing options
- training: optimizer, schedules, epochs, batch size
- distributed: backend, world_size, seed
- logging: checkpoints path, tensorboard settings

Example config snippet (YAML):

```yaml
model:
  name: resnet34
  num_classes: 10

dataset:
  name: cifar10
  input_size: 32

training:
  epochs: 100
  batch_size: 128
  optimizer:
    name: sgd
    lr: 0.1

distributed:
  backend: nccl
  seed: 42
```

## Checkpoints & Logging

- Checkpoints are written to `checkpoints/` by default. Ensure `checkpoints/` is listed in `.gitignore`.
- Logging integrates with TensorBoard; use `--log-dir` to change the destination.
- Save the full config file alongside logs and checkpoints for reproducibility.

## Reproducibility

To reproduce experiments reliably:

- Save the exact config file and CLI command used.
- Record environment details (Python version, package versions, CUDA/cuDNN).
- Pin dependencies in `requirements.txt` or use a Docker image.
- Optionally use deterministic seeds and document hardware used.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a feature branch
3. Add tests and documentation for significant changes
4. Open a pull request with a clear description of changes

Please follow the repository's code style and run existing tests before opening a PR.

## License

This project is provided under the MIT License. See `LICENSE` for details.

## Contact

If you have questions or suggestions, open an issue or contact the maintainer: https://github.com/pauldebojyoti
