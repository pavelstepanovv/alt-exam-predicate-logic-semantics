"""Скачивание базовой mT5-small фиксированной версии."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download():
    config = json.loads((ROOT / "models/model.json").read_text(encoding="utf-8"))
    directory = ROOT / config["base_directory"]
    expected = config["base_sha256"]
    if all((directory / name).is_file() and digest(directory / name) == checksum for name, checksum in expected.items()):
        print("Базовая модель уже скачана и проверена.")
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=config["model"], revision=config["revision"],
                      allow_patterns=list(expected), local_dir=directory)
    for name, checksum in expected.items():
        if digest(directory / name) != checksum:
            raise ValueError("Не совпадает контрольная сумма: " + name)
    print("Базовая модель скачана и проверена.")


if __name__ == "__main__":
    download()
