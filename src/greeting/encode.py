import typing
import numpy as np


class CharEncoder:
    def __init__(self):
        self._int_of_char: typing.Optional[dict[str, int]] = None
        self._char_of_int: typing.Optional[dict[int, str]] = None

    def _check_init(self):
        if self._int_of_char is None or self._char_of_int is None:
            raise RuntimeError("Call to fit() required.")

    def fit(self, data: str):
        chars = list(sorted(set(data)))

        self._char_of_int = dict(enumerate(chars))

        reverse_mapping = [(y, x) for (x, y) in self._char_of_int.items()]

        self._int_of_char = dict(reverse_mapping)

    def alphabet_size(self) -> int:
        self._check_init()
        assert self._int_of_char is not None
        return len(self._int_of_char)

    def encode(self, value: str) -> np.ndarray:
        self._check_init()
        assert self._int_of_char is not None
        assert self.alphabet_size() < 256
        return np.array([self._int_of_char[c] for c in value], dtype=np.uint8)

    def decode(self, value: np.ndarray) -> str:
        self._check_init()
        assert self._char_of_int is not None
        return "".join([self._char_of_int[i] for i in value])


def one_hot(data: np.ndarray, n_labels: int) -> np.ndarray:
    # Expect shaped encoded data
    assert len(data.shape) == 2

    retval_shape = data.shape + (n_labels,)

    # Initialize the the encoded array
    one_hot = np.zeros(retval_shape, dtype=np.float32)

    for idx, row in enumerate(data):
        for jdx, char in enumerate(row):
            one_hot[idx, jdx, char] = 1

    return one_hot
