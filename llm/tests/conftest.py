import os
import sys

_LLM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_LLM_ROOT, "data")

for path in (_LLM_ROOT, _DATA_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
