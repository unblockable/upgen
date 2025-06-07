from itertools import repeat
import typing

import numpy as np

from enum import Enum, auto
import torch
import torch.nn
from torch.distributions import Categorical

from encode import one_hot


class ModelType(Enum):
    RNN = auto()
    GRU = auto()
    LSTM = auto()


class SeqModel(torch.nn.Module):
    def __init__(
        self,
        model_type: ModelType,
        input_dim: int,
        hidden_dim=256,
        dropout_pr: float = 0.1,
        nlayers: int = 3,
    ):
        super().__init__()

        layer: typing.Any = None

        if model_type == ModelType.RNN:
            layer = torch.nn.RNN
        elif model_type == ModelType.GRU:
            layer = torch.nn.GRU
        elif model_type == ModelType.LSTM:
            layer = torch.nn.LSTM
        else:
            raise NotImplementedError()

        self.l = layer(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=nlayers,
            batch_first=True,
            dropout=dropout_pr,
        )

        self.dropout = torch.nn.Dropout(dropout_pr)

        self.linear = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x, _ = self.l(x)
        x = self.dropout(x)
        x = self.linear(x)

        if len(x.shape) == 3:
            return x[:, -1, :]
        else:
            return x[-1, :]

    def sample(self, start_token, pad_token, window_size, nlabels, dev, temp=1.0):
        initial_window = np.array(
            list(repeat(start_token, window_size)), dtype=np.uint8
        ).reshape(-1, window_size)

        one_hot_window = torch.tensor(one_hot(initial_window, nlabels)).to(dev)
        one_hot_window = one_hot_window.squeeze()

        output = []
        next_token = None
        ctr = 0

        while next_token is None or next_token != pad_token:
            if ctr > 100:
               # Try again.
                return self.sample(
                    start_token, pad_token, window_size, nlabels, dev, temp
                )

            next_char_prs = self(one_hot_window)

            dist = Categorical(logits=next_char_prs / temp)
            next_token = dist.sample().item()

            initial_window[0, 0 : (window_size - 1)] = initial_window[0, 1:]
            initial_window[0, -1] = next_token

            one_hot_window = torch.tensor(one_hot(initial_window, nlabels)).to(dev)
            one_hot_window = one_hot_window.squeeze()

            output.append(next_token)

            ctr += 1

        if output[0] == pad_token:
           # Try again.
            return self.sample(
                start_token, pad_token, window_size, nlabels, dev, temp
            )

        return output[:-1]
