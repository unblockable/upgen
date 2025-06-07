import argparse
import logging
import pickle
import typing
import sys

from model import SeqModel
from encode import CharEncoder as CharEncoder
import train
from train import Params as Params

import torch


def load_model(
    model_filepath: str,
    encoder_filepath: str,
    best_params_filepath: str,
    cuda: typing.Optional[int],
):
    if cuda is not None:
        dev = torch.device("cuda:0")
    else:
        dev = torch.device("cpu")

    with open(encoder_filepath, "rb") as in_f:
        encoder = pickle.load(in_f)

    with open(best_params_filepath, "rb") as in_f:
        params = pickle.load(in_f)

    model = SeqModel(
        model_type=params.model_type,
        input_dim=encoder.alphabet_size(),
        hidden_dim=params.hidden_dim,
        nlayers=params.nlayers,
    ).to(dev)

    if cuda is None:
        model.load_state_dict(
            torch.load(model_filepath, map_location=torch.device("cpu"))
        )
    else:
        model.load_state_dict(torch.load(model_filepath))

    model.eval()

    return model, encoder, params, dev


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

    model, encoder, params, dev = load_model(
        args.model_filepath, args.encoder_filepath, args.best_params_filepath, args.cuda
    )

    samples = predict(model, encoder, params, dev, args.nsamples, args.temp)

    for sample in samples:
        print(sample)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--log_level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )

    parser.add_argument("-c", "--cuda", type=int)

    parser.add_argument("best_params_filepath")
    parser.add_argument("encoder_filepath")
    parser.add_argument("model_filepath")
    parser.add_argument("-n", "--nsamples", default=1, type=int)
    parser.add_argument("-t", "--temp", default=1, type=float)
    return parser.parse_args()


def predict(
    model: SeqModel,
    encoder: CharEncoder,
    params: Params,
    dev,
    nsamples: int,
    temp: float,
):
    retval = []

    start_token = encoder.encode(train.START_CHAR)[0]
    pad_token = encoder.encode(train.PAD_CHAR)[0]

    with torch.no_grad():
        for _ in range(nsamples):
            s = model.sample(
                start_token,
                pad_token,
                params.window_size,
                encoder.alphabet_size(),
                dev,
                temp=temp,
            )

            retval.append(encoder.decode(s))

    return retval


if __name__ == "__main__":
    main(parse_args())
