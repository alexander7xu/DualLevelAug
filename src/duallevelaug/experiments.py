from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

DATASETS = ("scannet", "s3dis", "semantickitti")
CLASSES_SCANNET = ("wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower_curtain", "toilet", "sink", "bathtub", "otherfurniture")
RELATIONS = ("supported_by", "attached_to", "left_of", "right_of", "nearby", "faces", "oriented_with", "none")


def _condition(dataset, model="ptv3", variant="baseline", ratio=None, **extra):
    ours = variant == "ours"
    cfg = {"dataset": dataset, "model": model, "variant": variant,
           "ratio": (0.25 if ours else 0.0) if ratio is None else ratio,
           "seed": 0, "constraints": "full", "poisson": dataset in ("scannet", "s3dis"),
           "gt_sampling": dataset == "scannet", "protocol": "standard"}
    if model == "octformer":
        cfg["mix3d"] = variant == "mix3d"
    # PTv3 inherits its pinned backend's augmentation policy, including Mix3D.
    cfg.update(extra)
    return cfg


def catalog() -> list[dict[str, Any]]:
    result = []

    def add(identifier, config, metrics=("miou",), kind="evaluation", **kwargs):
        entry = {"id": identifier, "paper": identifier.split(".")[0], "kind": kind,
                 "config": config, "cells": [{"id": identifier + "." + key,
                   "metric": key, "label": key} for key in metrics],
                 "plot": "table" if kind != "figure" else identifier}
        entry.update(kwargs)
        result.append(entry)

    models = (("pointnext-xl", "baseline", DATASETS[:2]),
              ("minkunet", "baseline", DATASETS),
              ("sphereformer", "baseline", ("semantickitti",)),
              ("ptv2", "baseline", DATASETS), ("octformer", "baseline", DATASETS),
              ("octformer", "mix3d", DATASETS), ("ptv3", "baseline", DATASETS),
              ("octformer", "ours", DATASETS), ("ptv3", "ours", DATASETS))
    for model, variant, datasets in models:
        for dataset in datasets:
            add(f"table2.{model}.{variant}.{dataset}", _condition(dataset, model, variant))
    for dataset in ("s3dis", "semantickitti"):
        for constraints, ratio in (("random", .25), ("geometric", .25), ("local", .25), ("full", .1), ("full", .5), ("full", .25)):
            add(f"table3.{dataset}.{constraints}.{int(ratio * 100)}", _condition(dataset, variant="ours", ratio=ratio, constraints=constraints))
    for area in range(1, 7):
        for condition in ("before", "after", "generated"):
            if area == 5 and condition == "generated":
                continue
            add(f"table4.area{area}.{condition}", _condition("s3dis", variant="baseline" if condition == "before" else "ours", eval_area=area, eval_source="generated" if condition == "generated" else "original", eval_split="all"), ("allacc", "miou"))
    for dataset in DATASETS:
        add(f"table5.{dataset}", _condition(dataset), ("decomposition.floor", "decomposition.background", "decomposition.foreground"), kind="statistics")
    for protocol in ("synthetic_to_wmsc", "synthetic_in_domain"):
        for ratio in (0., .1, .25):
            add(f"table6.{protocol}.{int(ratio * 100)}", _condition("stpls3d", variant="ours" if ratio else "baseline", ratio=ratio, protocol={"synthetic_to_wmsc": "cross-domain", "synthetic_in_domain": "in-domain"}[protocol], eval_split="test"))
    for dataset in DATASETS:
        metrics = tuple(f"relationships.{relation}.{stat}" for relation in ("furniture_instances",) + RELATIONS for stat in ("total", "mean"))
        add(f"table7.{dataset}", _condition(dataset), metrics, kind="statistics")
    metrics = tuple("per_class_iou." + c for c in CLASSES_SCANNET) + ("miou",)
    for poisson in (False, True):
        add(f"table8.poisson_{str(poisson).lower()}", _condition("scannet", variant="ours", poisson=poisson), metrics)
    for sampling in (False, True):
        add(f"table9.gt_sampling_{str(sampling).lower()}", _condition("scannet", variant="ours", gt_sampling=sampling), metrics)
    figure_sources = {
        1: [], 2: [], 3: [],
        4: ["table2.octformer.ours.s3dis", "table2.octformer.ours.semantickitti"],
        5: ["table2.ptv3.ours.s3dis"], 6: ["table2.ptv3.ours.scannet"],
        7: ["table6.synthetic_to_wmsc.25"], 8: ["table7.scannet"],
        9: ["table8.poisson_true"], 10: ["table8.*"], 11: ["table9.*"]}
    for number, sources in figure_sources.items():
        add(f"figure{number}", {"figure": f"figure{number}", "seed": 0}, (), kind="figure", sources=sources)
    return result


def select(selectors=None):
    if isinstance(selectors, str):
        selectors = [x.strip() for x in selectors.split(",") if x.strip()]
    selectors = selectors or ["all"]
    entries = catalog()
    selected = []
    for selector in selectors:
        matches = [e for e in entries if selector == "all" or e["paper"] == selector or fnmatch.fnmatchcase(e["id"], selector)]
        if not matches:
            raise ValueError("Unknown experiment selector: " + selector)
        for entry in matches:
            if entry not in selected:
                selected.append(entry)
    return selected


def _digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def plan(selectors=None, root=".", overrides=None) -> list[dict[str, Any]]:
    """Build a dependency-ordered plan without loading data or running tasks.

    Per-dataset options in ``overrides.datasets`` override common options.
    """
    root = Path(root).resolve()
    overrides = dict(overrides or {})
    dataset_overrides = overrides.pop("datasets", {})
    nodes, targets = {}, {}

    def path(*parts):
        return str(root.joinpath(*parts))

    def node(stage, config, dependencies, output_names):
        key = stage + "-" + _digest(config)
        if key not in nodes:
            config = dict(config)
            output = config.setdefault("output", path("artifacts", "tasks", key))
            config_path = path("artifacts", "plans", key + ".json")
            nodes[key] = {"id": key, "stage": stage, "dependencies": list(dict.fromkeys(dependencies)),
                          "argv": [sys.executable, "-m", "duallevelaug", stage, "--config", config_path],
                          "config": config, "config_path": config_path,
                          "outputs": [str(Path(output) / name) for name in output_names]}
        return nodes[key]

    def preparation(cfg):
        dataset = cfg["dataset"]
        protocol = cfg.get("protocol", "standard")
        tag = dataset + ("-" + protocol if protocol != "standard" else "")
        shared = {"dataset": dataset, "protocol": protocol, "seed": cfg.get("seed", 0),
                  "raw_root": cfg.get("raw_root", path("data", "raw", dataset)),
                  "manifest": cfg.get("manifest", path("data", tag, "manifest.json")),
                  "output": cfg.get("prepare_output", path("data", tag))}
        for name in ("split_file", "label_map", "train_scenes", "test_scenes", "block_size", "block_stride"):
            if name in cfg:
                shared[name] = cfg[name]
        prep = None if cfg.get("manifest") else node("prepare", shared, [], ["manifest.json"])
        return shared, prep

    def training(cfg):
        cfg = {k: v for k, v in cfg.items() if not k.startswith("eval_")}
        shared, prep = preparation(cfg)
        cfg["manifest"] = shared["manifest"]
        cfg.pop("output", None)
        dependencies = [prep["id"]] if prep is not None else []
        generation = None
        if cfg.get("variant") == "ours":
            stat, pretrain, generation = augmentation(cfg, shared, prep)
            cfg["generated_manifest"] = generation["outputs"][0]
            dependencies.append(generation["id"])
        task = node("train", cfg, dependencies, ["model.pt"])
        return task, generation

    def statistics(cfg, shared, prep):
        stat_cfg = {k: shared[k] for k in ("dataset", "protocol", "seed", "manifest")}
        for name in ("relations", "relation_thresholds", "decomposition", "nearest_neighbors", "min_points", "component_radius"):
            if name in cfg:
                stat_cfg[name] = cfg[name]
        stat = node("stats", stat_cfg, [prep["id"]] if prep is not None else [], ["stats.json", "library.json", "graphs.json"])
        return stat

    def augmentation(cfg, shared, prep):
        stat = statistics(cfg, shared, prep)
        gen_cfg = {"dataset": cfg["dataset"], "protocol": cfg.get("protocol", "standard"),
                   "manifest": shared["manifest"], "stats": stat["outputs"][0],
                   "seed": cfg.get("seed", 0), "ratio": cfg.get("ratio", .25),
                   "constraints": cfg.get("constraints", "full"),
                   "poisson": cfg.get("poisson", False), "gt_sampling": cfg.get("gt_sampling", False)}
        for name in ("optimization", "poisson_options", "gt_sampling_classes", "gt_sampling_multiplier", "sampling", "graph", "candidates", "edge_noise", "cooccurrence_threshold", "steps", "pose_lr", "loss_weights", "relation_thresholds", "device", "layout_attempts", "collision_thickness"):
            if name in cfg:
                gen_cfg[name] = cfg[name]
        dependencies = [stat["id"]]
        pretrain = None
        if gen_cfg["constraints"] == "full":
            if cfg.get("graph_checkpoint"):
                gen_cfg["graph_checkpoint"] = cfg["graph_checkpoint"]
            else:
                graph_cfg = {"dataset": cfg["dataset"], "stats": stat["outputs"][0], "seed": cfg.get("seed", 0)}
                for graph_key in ("graph", "device"):
                    if graph_key in cfg:
                        graph_cfg[graph_key] = cfg[graph_key]
                pretrain = node("pretrain-graph", graph_cfg, [stat["id"]], ["graph.pt"])
                gen_cfg["graph_checkpoint"] = pretrain["outputs"][0]
                dependencies.append(pretrain["id"])
        return stat, pretrain, node("generate", gen_cfg, dependencies, ["manifest.json"])

    def resolve(entry):
        if entry["id"] in targets:
            return targets[entry["id"]]
        cfg = {**entry["config"], **overrides}
        cfg.update(dataset_overrides.get(cfg.get("dataset"), {}))
        if entry["kind"] == "figure":
            cfg = {**cfg, "root": str(root), "selector": entry["id"]}
            dependencies, sources = [], []
            for source in entry["sources"]:
                for original in select(source):
                    source_cfg = {**original["config"], **overrides}
                    source_cfg.update(dataset_overrides.get(source_cfg.get("dataset"), {}))
                    number = int(entry["id"].replace("figure", ""))
                    if number in (4, 6, 7, 9):
                        shared, prep = preparation(source_cfg)
                        stat, _, generated = augmentation(source_cfg, shared, prep)
                        dependencies.append(generated["id"])
                        sources.append({"dataset": source_cfg["dataset"], "manifest": shared["manifest"], "generated_manifest": generated["outputs"][0], "stats": stat["outputs"][0]})
                    elif number == 5:
                        shared, prep = preparation(source_cfg)
                        _, _, generated = augmentation(source_cfg, shared, prep)
                        extractor_cfg = _condition("s3dis", model="pointnet2")
                        extractor_cfg.update(overrides)
                        extractor_cfg.update(dataset_overrides.get("s3dis", {}))
                        trained, _ = training(extractor_cfg)
                        feature_cfg = {**extractor_cfg, "manifest": shared["manifest"], "generated_manifest": generated["outputs"][0], "checkpoint": trained["outputs"][0], "feature_layer": "final_sa_maxpool"}
                        feature = node("features", feature_cfg, [trained["id"], generated["id"]], ["features.npz"])
                        dependencies.append(feature["id"])
                        sources.append({"features": feature["outputs"][0]})
                    else:
                        task = resolve(original)
                        dependencies.append(task["id"])
                        sources.append({"experiment": original["id"], "result": task["outputs"][0], "config": task["config"]})
            cfg["sources"] = sources
            extension = ".md" if int(entry["id"].replace("figure", "")) <= 3 else ".png"
            result = node("report", cfg, dependencies, [entry["id"] + extension])
        elif entry["kind"] == "statistics":
            shared, prep = preparation(cfg)
            result = statistics(cfg, shared, prep)
        else:
            trained, generated = training(cfg)
            eval_cfg = dict(trained["config"])
            eval_cfg.pop("output", None)
            eval_cfg.update({k: v for k, v in cfg.items() if k.startswith("eval_")})
            eval_cfg["checkpoint"] = trained["outputs"][0]
            result = node("evaluate", eval_cfg, [trained["id"]], ["metrics.json"])
        result.setdefault("experiments", []).append(entry["id"])
        targets[entry["id"]] = result
        return result

    selected_entries = select(selectors)
    for entry in selected_entries:
        resolve(entry)
    table_entries = [entry for entry in selected_entries if entry["kind"] != "figure"]
    if table_entries:
        summary_config = {"root": str(root), "selectors": [entry["id"] for entry in table_entries],
                          "overrides": {**overrides, "datasets": dataset_overrides}}
        node("report", summary_config, [targets[entry["id"]]["id"] for entry in table_entries],
             ["report.json", "report.csv", "report.md"])
    visited = set()
    for task in nodes.values():
        if not set(task["dependencies"]).issubset(visited):
            raise RuntimeError("Invalid experiment DAG at " + task["id"])
        visited.add(task["id"])
    return list(nodes.values())


def _file_signature(path, content=True):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "missing": True}
    if path.is_dir():
        return {"path": str(path), "files": [_file_signature(p, content=False) for p in sorted(path.rglob("*")) if p.is_file()]}
    info = path.stat()
    result = {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    if content:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    if path.name in ("manifest.json", "library.json", "graphs.json"):
        document = json.loads(path.read_text())
        records = document.get("scenes", []) + document.get("objects", []) + document.get("backgrounds", []) + document.get("graphs", [])
        records += [component for background in document.get("backgrounds", []) for component in background.get("components", [])]
        result["child_signatures"] = [_file_signature(Path(row["path"]) if Path(row["path"]).is_absolute() else path.parent / row["path"], content=False) for row in records if "path" in row]
    return result


def _input_signatures(config, root="."):
    keys = {"manifest", "stats", "graph_checkpoint", "checkpoint", "generated_manifest", "raw_root", "features", "result", "split_file", "library", "graphs"}
    found = {}
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keys and isinstance(child, str):
                    input_path = Path(child)
                    if not input_path.is_absolute():
                        input_path = Path(root) / input_path
                    found[child] = _file_signature(input_path)
                    if key in ("manifest", "generated_manifest") and input_path.is_file():
                        manifest = json.loads(input_path.read_text())
                        scene_signatures = []
                        for row in manifest.get("scenes", []):
                            scene = Path(row["path"])
                            if not scene.is_absolute():
                                scene = input_path.parent / scene
                            scene_signatures.append(_file_signature(scene, content=False))
                        found[child]["scene_signatures"] = scene_signatures
                elif isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(config)
    return found


def run(selectors=None, root=".", overrides=None, resume=True, executor=None):
    """Resume uses size/mtime for raw data and referenced NPZ scenes to avoid
    hashing whole datasets; manifests, checkpoints and task outputs use SHA256.
    """
    executor = executor or subprocess.run
    results, receipts = [], {}
    for task in plan(selectors, root, overrides):
        config_path = Path(task["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(task["config"], ensure_ascii=False, indent=2) + "\n")
        receipt_path = config_path.with_suffix(".done.json")
        signature_config = dict(task["config"])
        if task["stage"] == "prepare":
            # manifest is this task's output, not an input to conversion.
            signature_config.pop("manifest", None)
        inputs = _input_signatures(signature_config, root)
        dependency_signatures = {identifier: _digest(receipts[identifier]) for identifier in task["dependencies"]}
        fingerprint = _digest({"config": task["config"], "inputs": inputs, "dependencies": dependency_signatures})
        old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        outputs = [_file_signature(path) for path in task["outputs"]]
        if resume and old.get("fingerprint") == fingerprint and old.get("output_signatures") == outputs and all(Path(p).exists() for p in task["outputs"]):
            receipts[task["id"]] = old
            results.append({"id": task["id"], "status": "reused"})
            continue
        executor(task["argv"], check=True, cwd=str(Path(root).resolve()))
        missing = [p for p in task["outputs"] if not Path(p).exists()]
        if missing:
            raise RuntimeError("Task finished without required outputs: " + ", ".join(missing))
        receipt = {"id": task["id"], "fingerprint": fingerprint, "inputs": inputs,
                   "dependency_signatures": dependency_signatures,
                   "output_signatures": [_file_signature(path) for path in task["outputs"]]}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        receipts[task["id"]] = receipt
        results.append({"id": task["id"], "status": "completed"})
    return results


def coverage():
    result = {}
    for entry in catalog():
        record = result.setdefault(entry["paper"], {"conditions": 0, "cells": 0})
        record["conditions"] += 1
        record["cells"] += len(entry["cells"])
    return result
