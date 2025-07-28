from importlib.resources import files
import json


def read_resource(name):
    if name not in {"backends.json", "paper_values.json"}:
        raise ValueError("Unknown packaged resource: " + name)
    return json.loads(files("duallevelaug.resources").joinpath(name).read_text(encoding="utf-8"))
