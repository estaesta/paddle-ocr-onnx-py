from __future__ import annotations

import os
from urllib.request import urlretrieve

from .types import ModelSource


CACHE_DIR = os.path.expanduser("~/.cache/ppu-paddle-ocr")


def resolve_source(value: str | None, err_msg: str) -> ModelSource:
    if not value:
        raise ValueError(err_msg)
    if value.lower().startswith("http"):
        return ModelSource(local_path=os.path.join(CACHE_DIR, os.path.basename(value)), url=value)
    return ModelSource(local_path=value, url=None)


def ensure_local_resource(source: ModelSource) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    if source.url and not os.path.exists(source.local_path):
        print(f"Downloading {source.url} -> {source.local_path}")
        urlretrieve(source.url, source.local_path)
        print(f"Downloaded: {source.local_path}")

    if not os.path.exists(source.local_path):
        raise FileNotFoundError(f"Required resource not found: {source.local_path}")
