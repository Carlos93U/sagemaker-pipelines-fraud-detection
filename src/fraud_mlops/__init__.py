"""Fraud MLOps — reusable package of the fraud-mlops project.

The notebook at the repo root imports this package; the logic lives here so it
can be reused (CLI, tests, another notebook) without copy-pasting cells.
"""

from . import config

__version__ = "1.0.0"
__all__ = ["config", "__version__"]
