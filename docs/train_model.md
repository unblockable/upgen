# Training a greeting string model

To run UPGen, you need a greeting string model. You can train one yourself, or
download a pretrained model. This document describes how to train a model
yourself if you want to reproduce the model, provide your own training data, or
tweak parameters.

Run these commands from the repository root. Generated data and model files
are kept in the `build/` directory.

In the UPGen USENIX Security publication, we used Github project names to train
the greeting string model. So, that's what is described below, but it should be
simple to substitute different sources of training data.

---

1. Download the training data set. The included command retrieves repository
   names from one hour of [GH Archive](https://www.gharchive.org/) data:

```
$ make download
```

By default, the script downloads the archive for January 1, 2015 at 15:00 UTC
and writes up to 10,000 names to `build/repos.txt`. Pass downloader options
through `ARGS`, such as `make download ARGS='--date 2025-01-01 --hour 12'`.

You can also tune and train from your own input file. The code assumes that it
contains lines following a `owner/repository` pattern. You can change the
specified training file by setting `REPOS_FILE` for both following commands.

2. Then, perform a hyperparameter search:

```
$ make tune
```

This step generates `build/best_params.pkl`. Without additional options,
training automatically selects Metal Performance Shaders on a supported Mac
and otherwise uses the CPU.

On Linux with a compatible NVIDIA GPU and driver, select the CUDA backend for
every command in the workflow:

```
$ make sync CUDA=0
$ make download CUDA=0
$ make tune CUDA=0
```

This command may take a while to complete.

3. Finally, train the model:

```
$ make train
```

This creates `build/model/encoder.pkl` and `build/model/model.torch`. For CUDA
training, run `make train CUDA=0`. Once training finishes, the model is ready
to generate greeting strings.

The paths can be customized with the `REPOS_FILE`, `BEST_PARAMS_FILE`, and
`MODEL_DIR` Make variables.
