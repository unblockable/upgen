import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import repeat
import logging
from pathlib import Path
import pickle
import typing
import sys

import random
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data

import torch.nn

from tqdm import tqdm

from encode import CharEncoder, one_hot
from model import SeqModel, ModelType

START_CHAR = "^"
PAD_CHAR = " "

HYPERPARAM_TUNE_NSAMPLES = 5000
HYPERPARAM_TUNE_EPOCHS = 2
HYPERPARAM_TUNE_TRIES = 50


def lines_of_file(input_filepath: str) -> list[str]:
    lines = []

    with open(input_filepath, "r") as in_f:
        for line in in_f:
            lines.append(line.strip())

    return lines


def names_of_lines(lines: list[str]) -> list[str]:
    return list(line.strip('"').split("/")[1] for line in lines)


def windows_of_data(
    data: np.ndarray, window_nelems: int, start_token, pad_token, dtype=np.uint8
) -> np.ndarray:
    assert len(data.shape) == 1
    assert window_nelems >= 2

    new_seq = (
        list(repeat(start_token, window_nelems - 1))
        + list(x for x in data if x != pad_token)
        + [pad_token]
    )

    windows = []

    nelems = len(new_seq)

    for idx in range(0, nelems - window_nelems + 1):
        window = new_seq[idx : idx + window_nelems]
        windows.append(np.array(window, dtype=dtype))

    return np.stack(windows)


def X_y_of_windows(data: np.ndarray) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # Assumes data is already windowed
    nelems = data.shape[0]

    X = data[0 : nelems - 1]
    y = data[1:nelems]

    index = X[:, -1, 0] != 1
    X = X[index]
    y = y[index]

    y = y[:, -1, :]

    return torch.from_numpy(X), torch.from_numpy(y)


@dataclass
class Params:
    model_type: ModelType
    hidden_dim: int
    dropout_pr: float
    nlayers: int
    window_size: int
    batch_size: int
    nepochs: int


@dataclass
class ParamGrid:
    model_types: list[ModelType]
    hidden_dim_vals: list[int]
    dropout_pr_vals: list[float]
    nlayers_vals: list[int]
    window_size_vals: list[int]
    batch_size_vals: list[int]

    @staticmethod
    def default():
        return ParamGrid(
            model_types=[ModelType.RNN, ModelType.GRU, ModelType.LSTM],
            hidden_dim_vals=[64, 128, 256, 512, 1024],
            dropout_pr_vals=[0.1, 0.2, 0.3, 0.4, 0.5],
            nlayers_vals=[1, 2, 3],
            window_size_vals=[2, 4, 6, 8, 10, 12],
            batch_size_vals=[2, 4, 8, 16, 32, 64, 128],
        )

    @staticmethod
    def sample_params(nepochs: int) -> Params:
        param_grid = ParamGrid.default()

        return Params(
            model_type=random.choice(param_grid.model_types),
            hidden_dim=random.choice(param_grid.hidden_dim_vals),
            dropout_pr=random.choice(param_grid.dropout_pr_vals),
            nlayers=random.choice(param_grid.nlayers_vals),
            window_size=random.choice(param_grid.window_size_vals),
            batch_size=random.choice(param_grid.batch_size_vals),
            nepochs=nepochs,
        )


def train_model(
    encoder,
    encoded_names,
    start_token,
    pad_token,
    params,
    dev,
    output_dirpath=None,
) -> tuple[SeqModel, float]:
    logging.info(f"Trying with these params: {params}")

    windows = [
        windows_of_data(
            en, params.window_size, start_token=start_token, pad_token=pad_token
        )
        for en in encoded_names
    ]

    combined_windows: np.ndarray = np.concatenate(windows)

    one_hot_windows = one_hot(combined_windows, encoder.alphabet_size())
    X, y = X_y_of_windows(one_hot_windows)

    dataset = data.TensorDataset(X, y)

    training_data, validation_data = data.random_split(dataset, [0.9, 0.1])

    loader = data.DataLoader(training_data, shuffle=True, batch_size=params.batch_size)

    # Just small enough to fit a batch into memory
    val_loader = data.DataLoader(validation_data, batch_size=1000)

    model = SeqModel(
        model_type=params.model_type,
        input_dim=encoder.alphabet_size(),
        hidden_dim=params.hidden_dim,
        nlayers=params.nlayers,
    ).to(dev)

    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    val_losses = []

    best_model = None
    best_val_loss = float("inf")

    epoch_progress = tqdm(range(params.nepochs))
    for epoch in epoch_progress:
        model.train()
        for X_batch, y_batch in tqdm(loader, leave=False):
            X_batch = X_batch.to(dev)
            y_batch = y_batch.to(dev)

            y_pred = model(X_batch)

            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_progress.set_postfix(epoch=epoch, loss=loss.item())

        model.eval()

        with torch.no_grad():
            val_loss = 0.0

            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(dev)
                y_batch = y_batch.to(dev)

                y_pred = model(X_batch)
                val_loss += loss_fn(y_pred, y_batch).item()

            logging.info(f"Total validation loss: {val_loss}")
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_model = deepcopy(model)
                best_val_loss = val_loss

                if output_dirpath is not None:
                    model_filepath = Path(output_dirpath) / Path("model.torch")
                    torch.save(best_model.state_dict(), model_filepath)

            distance = len(val_losses) - np.argmin(val_losses) - 1

            logging.info(f"Best model was found {distance} epochs ago...")

            if distance >= 5:
                logging.info("Early stopping condition hit!")
                break

            for _ in range(5):
                sample = model.sample(
                    start_token,
                    pad_token,
                    params.window_size,
                    encoder.alphabet_size(),
                    dev,
                )

                sample = encoder.decode(sample)
                logging.info(f"Sample:\t{sample}")

    assert best_model is not None
    return (best_model, best_val_loss)


def main(args: argparse.Namespace):
    log_level_of_str = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    format = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
    logging.basicConfig(
        level=log_level_of_str[args.log_level], format=format, stream=sys.stderr
    )
    logging.info(f"Program arguments:\t{args}")

    dev = None

    if args.cuda is not None:
        dev = torch.device("cuda:0")
    else:
        #dev = torch.device("cpu")
        dev = torch.device("mps")

    assert dev is not None

    names = names_of_lines(lines_of_file(args.input_filepath))

    if args.hyperparam_tune:
        names = random.sample(names, HYPERPARAM_TUNE_NSAMPLES)
    else:
        names = random.sample(names, 10000)

    name_lengths = [len(name) for name in names]

    logging.info(f"Average name length: {np.mean(name_lengths)}")

    longest_name_nchars = max(len(name) for name in names)

    logging.info(f"Longest name is {longest_name_nchars} chars long.")

    padded_names = [
        START_CHAR + name.ljust(longest_name_nchars + 1, PAD_CHAR) for name in names
    ]

    encoder = CharEncoder()
    encoder.fit(PAD_CHAR.join(padded_names))

    logging.info(encoder._char_of_int)

    if args.output_dirpath is not None:
        encoder_filepath = Path(args.output_dirpath) / Path("encoder.pkl")
        with open(encoder_filepath, "wb") as out_f:
            pickle.dump(encoder, out_f)


    encoded_names = list(encoder.encode(pn) for pn in padded_names)

    start_token = encoder.encode(START_CHAR)[0]
    pad_token = encoder.encode(PAD_CHAR)[0]

    assert pad_token == 0

    if args.hyperparam_tune:
        best_params = None
        best_val_loss = float("inf")

        for _ in range(HYPERPARAM_TUNE_TRIES):
            nepochs = HYPERPARAM_TUNE_EPOCHS

            params = ParamGrid.sample_params(nepochs)

            model, val_loss = train_model(
                encoder, encoded_names, start_token, pad_token, params, dev
            )

            if val_loss < best_val_loss:
                logging.info(f"New best params: {params}\tval loss: {val_loss}")
                best_params = deepcopy(params)
                best_val_loss = val_loss

        with open(args.output_filepath, "wb") as out_f:
            pickle.dump(best_params, out_f)
    else:
        assert args.best_params_filepath is not None

        with open(args.best_params_filepath, "rb") as in_f:
            best_params = pickle.load(in_f)

        assert best_params is not None
        best_params.nepochs = 100

        model, val_loss = train_model(
            encoder,
            encoded_names,
            start_token,
            pad_token,
            best_params,
            dev,
            args.output_dirpath,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    parser.add_argument("-b", "--best_params_filepath")
    parser.add_argument("-d", "--output_dirpath")
    parser.add_argument("-c", "--cuda", type=int)
    parser.add_argument("-y", "--hyperparam_tune", action="store_true")
    parser.add_argument("input_filepath")
    parser.add_argument("--output_filepath")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
