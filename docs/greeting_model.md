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

!! TODO: Add download instructions

The release bundle will be packaged so that extracting it into `build/` creates
the directory layout shown above.

## Train a model locally

See the training documentation: [train_model.md].
