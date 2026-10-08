# CLI

This document explains the CLI behaviour of UPGen.

---

## train.py

```
python train.py [options] input_filepath
```

### Arguements

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
| --output_filepath PATH            | Save hyperparameter-tuning results to PATH, normally its best_params.pkl.  |

---

## generate.py

```
python generate.py [options] \
    config_filepath \
    best_params_filepath \
    encoder_filepath \
    model_filepath
```

### Arguements

| Argument               | Description              |
| ---------------------- | ------------------------ |
| config_filepath        | Path to config.json.     |
| best_params_filepath   | Path to best_params.pkl. |
| encoder_filepath       | Path to encoder.pkl.     |
| model_filepath         | Path to model.torch.     |

### Flags

| Option                          | Description                                                         |
| ------------------------------- | ------------------------------------------------------------------- |
| -h, --help                      | Show the help message.                                              |
| -l, --log_level                 | Set logging verbosity to debug, info, warning, error, or critical.  |
| -s, --seed SEED                 | Set the random seed used for protocol-parameter sampling.           |
| -o, --output_filepath PATH      | Write PSFs to PATH.                                                 |
| -t, --greeting_string_temp TEMP | Set the sampling temperature used by the greeting-string model.     |
| -n, --num_generated NUMBER      | Number of PSFs to generate.                                         |
| -b, --best                      | Generate the fixed best-case.                                       |
| -w, --worst                     | Generate the fixed worst-case.                                      |

---
