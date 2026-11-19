"""Central configuration for data location.

By default, data is expected in the ./data folder next to this file. Override with the
OULAD_DATA environment variable, e.g.:

    OULAD_DATA=/path/to/oulad python run_all.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("OULAD_DATA") or os.path.join(HERE, "data")
