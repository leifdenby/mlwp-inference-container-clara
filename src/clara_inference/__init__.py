"""Inference for CLARA (CRPS-LAM Rapid Atmospheric ensembles), the CRPS-LAM
probabilistic ML limited-area weather model."""
import importlib.metadata

try:
    __version__ = importlib.metadata.version("clara-inference")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
