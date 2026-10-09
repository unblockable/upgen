# CLI

This document explains the CLI behaviour of UPGen.

The Make targets provide the standard project workflow and place their outputs
under the `build/` directory. The commands below document the underlying CLI for
custom usage.

---

## Optional model training

```
uv run upgen-train [options] input_filepath
```

The default backend group provides CPU-only PyTorch on Linux and MPS support on
macOS. For CUDA training on Linux, run uv with
`--no-group cpu-mps --group cuda` and pass `--cuda NUMBER`. The two backend
groups cannot be enabled together. Training is only needed when you are not
using a compatible pretrained greeting model.

### Arguments

| Argument               | Description              |
| ---------------------- | ------------------------ |
| input_filepath         | Path to input file.      |

### Flags

| Option                            | Description                                                                |
| --------------------------------- | -------------------------------------------------------------------------- |
| -h, --help                        | Show the help message.                                                     |
| -l, --log_level                   | Set logging verbosity to debug, info, warning, error, or critical.         |
| -b, --best_params_filepath PATH   | Load previously selected hyperparameters from PATH.                        |
| -d, --output_dirpath PATH         | Directory where trained model artifacts are saved.                         |
| -c, --cuda NUMBER                 | Enable CUDA training.                                                      |
| -y, --hyperparam_tune             | Run hyperparameter tuning.                                                 |
| --output_filepath PATH            | Save hyperparameter-tuning results to PATH, normally `build/best_params.pkl` in the Make workflow. |

---

## PSF generation

```
uv run upgen-generate [options] \
    config_filepath \
    best_params_filepath \
    encoder_filepath \
    model_filepath
```

The three model arguments may come from either a pretrained release or local
training. The configuration is passed separately so it can be changed without
modifying the installed package. See the [greeting string model
guide](greeting_model.md).

### Arguments

| Argument               | Description                                    |
| ---------------------- | ---------------------------------------------- |
| config_filepath        | Path to the protocol generation configuration. |
| best_params_filepath   | Path to best_params.pkl.                       |
| encoder_filepath       | Path to encoder.pkl.                           |
| model_filepath         | Path to model.torch.                           |

### Flags

| Option                          | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| -h, --help                      | Show the help message.                                              |
| -l, --log_level                 | Set logging verbosity to debug, info, warning, error, or critical.  |
| -s, --seed SEED                 | Set the random seed used for protocol-parameter sampling.           |
| -c, --cuda NUMBER               | Generate with the selected CUDA device.                             |
| -o, --output_filepath PATH      | Write PSFs to PATH; the Make workflow uses `build/upgen.psf`.       |
| -t, --greeting_string_temp TEMP | Set the sampling temperature used by the greeting-string model.     |
| -n, --num_generated NUMBER      | Number of PSFs to generate.                                         |
| -b, --best                      | Generate the fixed best-case.                                       |
| -w, --worst                     | Generate the fixed worst-case.                                      |

---

## Download optional training data

```
uv run upgen-download-repos [options]
```

| Option          | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| -h, --help      | Show the help message.                                         |
| --date DATE     | Select the GH Archive date in `YYYY-MM-DD` format.              |
| --hour HOUR     | Select the archive hour from 0 through 23.                      |
| --limit NUMBER  | Set the maximum number of repository names to write.            |
| --output PATH   | Write repository names to PATH; the Make workflow uses `build/repos.txt`. |

---
