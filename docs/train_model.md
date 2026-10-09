# Model Training

Before creating a PSF, we first need a model to generate greeting strings.
Run these commands from the repository root. Generated data and model artifacts
are kept under the ignored `build/` directory.

First, install the optional NumPy, PyTorch, and tqdm dependencies:

```
$ make sync-model
```

1. Generate the training input. The included command retrieves repository
   names from one hour of [GH Archive](https://www.gharchive.org/) data:

```
$ make download
```

By default, the script downloads the archive for January 1, 2015 at 15:00 UTC
and writes up to 10,000 names to `build/repos.txt`. Pass downloader options
through `ARGS`, such as `make download ARGS='--date 2025-01-01 --hour 12'`.
You can also tune and train from your own input file, provided that it contains
one `owner/repository` entry per line, by setting `REPOS_FILE` for both commands.

2. Find the best parameters for the model:

```
$ make tune
```

This step generates `build/best_params.pkl`. To select CUDA device 0 on a
machine with an NVIDIA GPU, run `make tune ARGS='--cuda 0'`. Without that
option, training automatically selects Metal Performance Shaders on a
supported Mac and otherwise uses the CPU.

This command may take a while to complete.

3. Train the model:

```
$ make train
```

This creates `build/model/encoder.pkl` and `build/model/model.torch`. As with
tuning, CUDA can be selected with `make train ARGS='--cuda 0'`. Once training
finishes, the model is ready to generate PSFs.

The paths can be customized with the `REPOS_FILE`, `BEST_PARAMS_FILE`, and
`MODEL_DIR` Make variables.
