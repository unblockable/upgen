"""Console entry points for model-backed commands."""

from collections.abc import Callable
from importlib import import_module


def model_command(module_name: str) -> Callable[[], None]:
    try:
        module = import_module(module_name, package=__package__)
    except ModuleNotFoundError as error:
        if error.name in {"numpy", "torch", "tqdm"}:
            raise SystemExit(
                "Project dependencies are not installed. Run `uv sync` for "
                "CPU/MPS, or `make sync CUDA=N` for CUDA device N."
            ) from error
        raise

    return module.cli


def generate() -> None:
    model_command(".generate")()


def predict() -> None:
    model_command(".greeting.predict")()


def train() -> None:
    model_command(".greeting.train")()
