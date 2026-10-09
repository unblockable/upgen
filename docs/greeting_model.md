# Greeting string models

UPGen uses a character-level language model to create greeting strings for
generated PSFs. The generator needs three model files:

```text
build/
├── best_params.pkl
└── model/
    ├── encoder.pkl
    └── model.torch
```

Training this model yourself is optional. You can either download a pretrained
model or [train one yourself](train_model.md).

## Use the pretrained model

You can download and use an example model from GitHub:

```console
$ make download-model
```

The Make target downloads
[`upgen_example_greeting_string_model_v0.1.0.tar.gz`](https://github.com/unblockable/upgen/releases/download/v0.1.0/upgen_example_greeting_string_model_v0.1.0.tar.gz)
into `build/`, extracts it there, and verifies that all three required files are
present. Use `make -B download-model` to download a new copy.

The default release tag is `v0.1.0`. A different published model or release tag
can be selected without editing the Makefile, for example:

```console
$ make download-model MODEL_RELEASE_TAG=greeting-model-v0.1.0
$ make download-model MODEL_URL=https://example.org/another-model.tar.gz
```

## Train a model locally

See the training documentation: [train_model.md].
