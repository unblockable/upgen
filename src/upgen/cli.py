"""Console entry points for commands with optional model dependencies."""

from collections.abc import Callable
from importlib import import_module


def model_command(module_name: str) -> Callable[[], None]:
    try:
        module = import_module(module_name, package=__package__)
    except ModuleNotFoundError as error:
        if error.name in {"numpy", "torch", "tqdm"}:
            raise SystemExit(
                "Model dependencies are not installed. "
                "Install them with `uv sync --extra model` for CPU/MPS or "
                "`uv sync --extra model-cuda` for CUDA."
            ) from error
        raise

    return module.cli


def generate() -> None:
    model_command(".generate")()


def predict() -> None:
    model_command(".greeting.predict")()


def train() -> None:
    model_command(".greeting.train")()
