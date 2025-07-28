import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import tempfile


def require(name):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError("Missing dependency %s. Install the appropriate project dependencies or optional extra; see README.md. Original error: %s" % (name, exc)) from exc


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np = require("numpy")
    np.random.seed(seed)
    return np.random.default_rng(seed)


def snapshot(output, cfg, **extra):
    from . import __version__
    from .resources import read_resource
    record = {"version": __version__, "config": cfg, "seed": cfg.get("seed", 0), **extra}
    for key in ("manifest", "generated_manifest", "stats", "graph_checkpoint", "checkpoint"):
        path = cfg.get(key)
        if path and Path(path).is_file():
            record[key + "_sha256"] = file_hash(path)
    record["upstream_versions"] = read_resource("backends.json")
    write_json(Path(output) / "provenance.json", record)
    write_json(Path(output) / "config.json", cfg)
    return record
