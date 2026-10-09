import importlib

import pytest


@pytest.mark.parametrize(
    "module", ["wzc", "cv2", "numpy", "ultralytics", "pydantic", "yaml", "fastapi"]
)
def test_import(module):
    importlib.import_module(module)
