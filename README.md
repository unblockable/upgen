# UPGen: The Unidentified Protocol Generator

## Synopsis

To generate a new PSF, run the following command:

```
$ python3 generate.py
    [-h]
    [-l {debug,info,warning,error,critical}]
    [-s SEED]
    [-o OUTPUT_FILEPATH]
    [-t GREETING_STRING_TEMP]
    [-n NUM_GENERATED]
    [-b]
    [-w]
    config_filepath best_params_filepath encoder_filepath model_filepath
```

Generation requires a trained greeting string generator. To train a greeting
string model, the following commands can be used:

1. Determine model hyperparameters

```
$ python3 train.py -y <DATA_FILEPATH> --output_filepath best_params.pkl
```

2. Train a model

```
$ python3 train.py -b best_params.pkl -d trained_model/ <DATA_FILEPATH>
```

## Research citation

You can read more about UPGen in our USENIX Security publication:

```
@inproceedings{wails:usenixsec:25,
  author       = {Ryan Wails
                  Rob Jansen and
                  Aaron Johnson and
                  Micah Sherr},
  editor       = {Lujo Bauer and
                  Giancarlo Pellegrino},
  title        = {Censorship Evasion with Unidentified Protocol Generation},
  booktitle    = {Proceedings of the 34th USENIX Security Symposium},
  publisher    = {{USENIX} Association},
  year         = {2025},
}
```
