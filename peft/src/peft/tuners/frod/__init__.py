from .layer import FRODLayer, Linear
from .model import FRODModel
from .config import FRODConfig

__all__ = ["Linear", "FRODLayer", "FRODModel", "FRODConfig"]

def __getattr__(name):

    raise AttributeError(f"module {__name__} has no attribute {name}")