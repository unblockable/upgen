# Installation

## Procedure

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
   clone the repository:

```
$ git clone https://github.com/unblockable/upgen.git
$ cd upgen
```

2. Create the project environment and install UPGen with its dependencies:

```
$ uv sync
```

   The equivalent Make command is `make sync`. PyTorch, NumPy, and tqdm are
   standard project dependencies. Both pretrained and locally trained models
   need PyTorch at runtime, although training the model itself remains optional.

   Plain `uv sync` selects CPU-only PyTorch on Linux and MPS-capable PyTorch on
   macOS. On Linux with a compatible NVIDIA GPU and driver, select the CUDA 13.0
   backend group instead:

```
$ uv sync --no-group cpu-mps --group cuda
```

   The equivalent Make command is `make sync CUDA=0`, where `0` is the CUDA
   device number to use in subsequent model commands. The `cpu-mps` and `cuda`
   backend groups are mutually exclusive. See the [greeting string model
   guide](greeting_model.md) for details on how to train a greeting string
   model.
