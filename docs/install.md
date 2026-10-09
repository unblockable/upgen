# Installation

## Procedure

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
   clone the repository:

```
$ git clone https://github.com/unblockable/upgen.git
$ cd upgen
```

2. Create the project environment and install the locked dependencies:

```
$ uv sync
```

   The equivalent Make command is `make sync`.

   This installs the basic UPGen source. To train and use greeting string
   language models, include the optional model dependencies:

```
$ uv sync --extra model
```

   The equivalent Make command is `make sync-model`.

   This selects CPU-only PyTorch on Linux and MPS-capable PyTorch on macOS. On
   Linux with a compatible NVIDIA GPU and driver, select the CUDA 13.0 build
   instead:

```
$ uv sync --extra model-cuda
```

   The equivalent Make command is
   `make sync-model MODEL_EXTRA=model-cuda`. The `model` and `model-cuda`
   extras conflict by design and must not be enabled together.
