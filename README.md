# S4-PTD

This is the code repository accompanying the manuscript ''There is HOPE to Avoid HiPPOs for Long-memory State Space Models." The repository is heavily adapted from the ''state-spaces" GitHub repository (https://github.com/HazyResearch/state-spaces.git). While it contains references to existing papers and code repositories, it includes no information that reveals the identities of the manuscript authors.

## Setup

### Requirements
This repository requires Python 3.9+ and Pytorch 1.10+.
It has been tested up to Pytorch 1.13.1.
Other packages are listed in [requirements.txt](./requirements.txt).
Some care may be needed to make some of the library versions compatible, particularly torch/torchvision/torchaudio/torchtext.

Example installation:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data

Basic datasets are auto-downloaded, including MNIST, CIFAR, and Speech Commands.
All logic for creating and loading datasets is in [src/dataloaders](./src/dataloaders/) directory.
The README inside this subdirectory documents how to download and organize other datasets.

### WandB

Logging with [WandB](https://wandb.ai/site) is built into this repository.
In order to use this, simply set your `WANDB_API_KEY` environment variable, and change the `wandb.project` attribute of [configs/config.yaml](configs/config.yaml) (or pass it on the command line e.g. `python -m train .... wandb.project=HOPE`).

Set `wandb=null` to turn off WandB logging.

## Execution

### Noisy sCIFAR-10

The scripts [./noisycifar_HOPE.py](./noisycifar_HOPE.py) and [./noisycifar_S4D.py](./noisycifar_S4D.py) contains enough information to reproduce the experiment on the noise-padded sCIFAR-10 problem.

### LRA Benchmarks

The Long-Range Arena benchmarks can be tested by running the bash scripts
```
run_foo.sh
```
where `foo` is the name of the problem, choosing from `listops`, `imdb`, `aan`, `cifar`, `pathfinder`, and `pathx`. So far, these experiments are not built upon the same pipeline. The three experiments on `cifar`, `pathfinder`, and `pathx` use a simplified pipeline with no lightning package. This slightly impairs the efficiency of the training.


