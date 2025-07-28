from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import experiments


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _class_key(value):
    return str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("other_furniture", "otherfurniture")


def computed_metrics(payload):
    """Return percentages, excluding zero-union classes from mean IoU."""
    matrix = payload.get("confusion_matrix", payload.get("confusion"))
    if matrix is None:
        raise ValueError("Evaluation result requires a confusion matrix")
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("Confusion matrix must be nonempty and square")
    if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0 for row in matrix for x in row):
        raise ValueError("Confusion matrix contains invalid counts")
    counts = [sum(row) for row in matrix]
    cols = [sum(row[j] for row in matrix) for j in range(n)]
    union = [counts[i] + cols[i] - matrix[i][i] for i in range(n)]
    iou = [100 * matrix[i][i] / union[i] if union[i] else None for i in range(n)]
    valid = [v for v in iou if v is not None]
    total = sum(counts)
    names = payload.get("class_names", [str(i) for i in range(n)])
    if len(names) != n:
        raise ValueError("Class names and confusion matrix dimensions differ")
    return {"miou": sum(valid) / len(valid) if valid else None,
            "allacc": 100 * sum(matrix[i][i] for i in range(n)) / total if total else None,
            "per_class_iou": {_class_key(name): value for name, value in zip(names, iou)},
            "unit": "percent", "absent_class_policy": "exclude zero union"}


def _lookup(payload, key):
    for component in key.split("."):
        if not isinstance(payload, dict) or component not in payload:
            return None
        payload = payload[component]
    return payload


def table_report(config):
    root = Path(config.get("root", ".")).resolve()
    output = Path(config.get("output", root / "artifacts" / "reports"))
    output.mkdir(parents=True, exist_ok=True)
    selectors = config.get("selectors", config.get("selector", "all"))
    entries = [e for e in experiments.select(selectors) if e["kind"] != "figure"]
    tasks = experiments.plan([e["id"] for e in entries], root, config.get("overrides")) if entries else []
    result_paths = {name: task["outputs"][0] for task in tasks for name in task.get("experiments", [])}
    result_paths.update(config.get("results", {}))
    from .resources import read_resource
    references = (_read(config["reference"]) if config.get("reference")
                  else read_resource("paper_values.json")).get("cells", {})
    rows = []
    for entry in entries:
        path = result_paths.get(entry["id"])
        payload, error = {}, None
        if path and Path(path).is_file():
            try:
                raw = _read(path)
                if entry["kind"] == "statistics":
                    payload = raw
                else:
                    if "class_names" not in raw:
                        from .data import DATASET_CLASSES
                        raw["class_names"] = DATASET_CLASSES[entry["config"]["dataset"]]
                    payload = computed_metrics(raw)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        for cell in entry["cells"]:
            value = _lookup(payload, cell["metric"])
            reference = references.get(cell["id"])
            status = "invalid" if error else "measured" if value is not None else "missing"
            rows.append({"cell": cell["id"], "experiment": entry["id"], "metric": cell["metric"],
                         "measured": value, "paper_reference": reference,
                         "difference": value - reference if value is not None and reference is not None else None,
                         "status": status, "result_file": path, "error": error})
    report = {"source": "actual confusion matrices and training-only component statistics",
              "paper_reference_is_not_result": True, "coverage": experiments.coverage(), "rows": rows,
              "measured_cells": sum(row["status"] == "measured" for row in rows),
              "missing_cells": sum(row["status"] == "missing" for row in rows),
              "invalid_cells": sum(row["status"] == "invalid" for row in rows)}
    _write(output / "report.json", report)
    with (output / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell", "experiment", "metric", "measured", "paper_reference", "difference", "status", "result_file", "error"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# DualLevelAug Experiment Report", "", "Measured values use confusion matrices and collected statistics. Published values are listed for comparison; unavailable results are marked missing.", "", "| Cell | Measured | Paper reference | Difference | Status |", "|---|---:|---:|---:|---|"]
    def fmt(value):
        return "missing" if value is None else f"{value:.4f}"
    for row in rows:
        lines.append("| " + " | ".join([row["cell"], fmt(row["measured"]), fmt(row["paper_reference"]), fmt(row["difference"]), row["status"]]) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _plot_dependencies():
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Figure plotting requires NumPy and Matplotlib; install the project's plots extra as described in README.md.") from exc
    return np, plt


def _scene(manifest_path, seed=0, generated=False):
    from .data import load_manifest, load_scene
    manifest = load_manifest(manifest_path)
    rows = [r for r in manifest["scenes"] if r.get("split") == "train" and (not generated or r.get("generated", True))]
    if not rows:
        raise ValueError("No training scene available in " + str(manifest_path))
    rows.sort(key=lambda r: r["id"])
    row = rows[seed % len(rows)]
    return row, load_scene(row["path"])


def _scatter(ax, scene, title, seed=0, max_points=50000, highlight=None):
    import numpy as np
    xyz = np.asarray(scene["coord"])
    if not len(xyz):
        raise ValueError("Cannot plot empty point cloud")
    indices = np.random.default_rng(seed).choice(len(xyz), min(len(xyz), max_points), replace=False)
    colors = np.asarray(scene.get("segment", np.zeros(len(xyz))))[indices]
    ax.scatter(*xyz[indices].T, c=colors, cmap="tab20", s=.3, rasterized=True)
    if highlight is not None and len(highlight):
        points = np.asarray(highlight)
        ax.scatter(*points.T, c="black", s=2, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    spans = np.maximum(np.ptp(xyz, axis=0), .1)
    ax.set_box_aspect(spans)
    ax.view_init(elev=25, azim=-60)


def _draw_org(ax, graph, title):
    import numpy as np
    count = len(graph["classes"])
    if count > 80:
        ax.text(.5, .5, f"{count} nodes; graph saved in source artifact", ha="center", va="center")
        ax.set_title(title); ax.axis("off")
        return
    angle = np.arange(count) * 2 * np.pi / max(count, 1)
    xy = np.column_stack((np.cos(angle), np.sin(angle)))
    edges = np.asarray(graph["edges"])
    for i in range(count):
        for j in range(count):
            if i != j and np.any(edges[i, j, :7]):
                ax.annotate("", xy=xy[j], xytext=xy[i], arrowprops={"arrowstyle": "->", "lw": .4, "alpha": .25})
    ax.scatter(xy[:, 0], xy[:, 1], c=graph["classes"], cmap="tab20", s=70)
    for i, point in enumerate(xy):
        ax.text(*point, str(i), fontsize=6, ha="center", va="center")
    ax.set_title(title); ax.axis("off")


def _tsne(config, output):
    np, plt = _plot_dependencies()
    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Figure 5 requires scikit-learn") from exc
    paths = [source["features"] for source in config["sources"]]
    features, sources, ids = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            features.append(data["features"])
            sources.extend(data["sources"].astype(str).tolist())
            ids.extend(data["scene_ids"].astype(str).tolist())
    values = np.concatenate(features)
    if len(values) < 4:
        raise ValueError("t-SNE requires at least four scene descriptors")
    if set(sources) != {"original", "generated"}:
        raise ValueError("t-SNE comparison requires original and generated descriptors")
    # A joint fit puts both descriptor groups in the same coordinate system.
    scaled = StandardScaler().fit_transform(values)
    seed = int(config.get("seed", 0))
    embedded = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=min(float(config.get("perplexity", 30)), len(values) - 1)).fit_transform(scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = np.asarray(sources)
    for source, cmap in (("original", "Blues"), ("generated", "Oranges")):
        points = embedded[groups == source]
        # Neighbor-radius density handles collinear groups where KDE is singular.
        if len(points) > 1:
            from sklearn.neighbors import NearestNeighbors
            distances, _ = NearestNeighbors(n_neighbors=min(8, len(points))).fit(points).kneighbors(points)
            density = 1 / np.maximum(distances[:, -1], 1e-6)
            density = .25 + .75 * (density - density.min()) / max(float(np.ptp(density)), 1e-6)
        else:
            density = np.ones(len(points))
        ax.scatter(*points.T, c=density, cmap=cmap, vmin=0, vmax=1, s=16, label=source)
    ax.legend(); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title("PointNet++ final-SA scene descriptors")
    fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)
    np.savez_compressed(output.with_suffix(".npz"), embedding=embedded, sources=groups, scene_ids=np.asarray(ids), seed=seed)


def _class_ablation(config, output):
    np, plt = _plot_dependencies()
    sources = config["sources"]
    payloads = []
    for source in sources:
        raw = _read(source["result"])
        raw.setdefault("class_names", experiments.CLASSES_SCANNET)
        payloads.append(computed_metrics(raw))
    names = list(experiments.CLASSES_SCANNET) + ["overall"]
    width = .8 / max(len(payloads), 1)
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (source, metrics) in enumerate(zip(sources, payloads)):
        values = [metrics["per_class_iou"].get(name) for name in names[:-1]] + [metrics["miou"]]
        ax.bar(np.arange(len(names)) + (i - (len(payloads) - 1) / 2) * width,
               [np.nan if value is None else value for value in values], width=width,
               label=source["experiment"].split(".")[-1])
    ax.set_xticks(np.arange(len(names)), names, rotation=60, ha="right")
    ax.set_ylabel("IoU (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(output, dpi=200); plt.close(fig)


def _scene_comparison(config, output):
    np, plt = _plot_dependencies()
    sources = config["sources"]
    number = int(config["figure"].replace("figure", ""))
    columns = 3 if number == 4 else 2
    fig = plt.figure(figsize=(columns * 5, len(sources) * 4))
    seed = int(config.get("seed", 0))
    recorded = []
    for i, source in enumerate(sources):
        row, original = _scene(source["manifest"], seed)
        generated_row, generated = _scene(source["generated_manifest"], seed, True)
        from .data import load_manifest, load_scene
        background_id = generated_row.get("background_scene", generated_row.get("source_id"))
        match = [r for r in load_manifest(source["manifest"])["scenes"] if r["id"] == background_id]
        if match:
            row, original = match[0], load_scene(match[0]["path"])
        panels = [(original, "Original: " + row["id"])]
        if number == 4:
            second_row, second = _scene(source["manifest"], seed + 1)
            mixed = {"coord": np.concatenate((original["coord"] - original["coord"].mean(0), second["coord"] - second["coord"].mean(0))),
                     "segment": np.concatenate((original["segment"], second["segment"]))}
            panels.append((mixed, "Mix3D illustration (centered scene union)"))
            recorded.append({"mix3d_sources": [row["id"], second_row["id"]]})
        panels.append((generated, "Generated: " + generated_row["id"]))
        for j, (scene, title) in enumerate(panels):
            ax = fig.add_subplot(len(sources), columns, i * columns + j + 1, projection="3d")
            _scatter(ax, scene, title, seed, int(config.get("max_points", 50000)))
        recorded.append({"dataset": source["dataset"], "original": row["id"], "generated": generated_row["id"]})
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
    _write(output.with_suffix(".sources.json"), recorded)


def _none_instance(config, output):
    np, plt = _plot_dependencies()
    source = config["sources"][0]
    stats = _read(source["result"])
    stat_dir = Path(source["result"]).parent
    graphs = _read(stats.get("graphs", stat_dir / "graphs.json"))["graphs"]
    library = _read(stats.get("library", stat_dir / "library.json"))
    from .data import load_manifest, load_scene
    manifest = load_manifest(stats.get("manifest", source["config"]["manifest"]))
    for graph in graphs:
        if "path" in graph:
            graph_path = Path(graph["path"])
            if not graph_path.is_absolute():
                graph_path = stat_dir / graph_path
            with np.load(graph_path, allow_pickle=False) as sparse:
                classes = sparse["classes"]
                edge_index = sparse["edge_index"]
                edge_values = sparse["edge_values"]
            outgoing = lambda index: edge_values[edge_index[:, 0] == index, :7].any()
        else:
            classes = graph["classes"]
            edges = np.asarray(graph["edges"])
            outgoing = lambda index: edges[index, :, :7].any()
        for index in range(int(graph.get("dynamic_count", len(classes)))):
            if outgoing(index):
                continue
            objects = [obj for obj in library["objects"] if obj.get("source_scene", obj.get("source_id")) == graph["scene_id"]]
            if graph.get("object_paths"):
                by_path = {obj["path"]: obj for obj in objects}
                objects = [by_path[path] for path in graph["object_paths"]]
            if index >= len(objects):
                continue
            pointcloud = load_scene(objects[index]["path"])
            # Library points are already in the source scene's coordinate frame.
            highlight = pointcloud["coord"]
            rows = [r for r in manifest["scenes"] if r["id"] == graph["scene_id"]]
            if not rows:
                continue
            scene = load_scene(rows[0]["path"])
            fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(111, projection="3d")
            _scatter(ax, scene, "No outgoing named relation (black instance)", int(config.get("seed", 0)), highlight=highlight)
            fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)
            _write(output.with_suffix(".sources.json"), {"scene_id": graph["scene_id"], "node_index": index, "object": objects[index]})
            return
    raise ValueError("No instance without named outgoing relationships was found in the training graphs")


def _completion(config, output):
    np, plt = _plot_dependencies()
    source = config["sources"][0]
    generated_row, generated = _scene(source["generated_manifest"], int(config.get("seed", 0)), True)
    from .data import load_manifest, load_scene
    rows = load_manifest(source["manifest"])["scenes"]
    background_id = generated_row.get("background_scene", generated_row.get("source_id"))
    matches = [row for row in rows if row["id"] == background_id]
    if not matches:
        raise ValueError("Generated scene provenance must identify background_scene for completion comparison")
    original = load_scene(matches[0]["path"])
    boundary = np.isin(original["segment"], [0, 1])
    before = {"coord": original["coord"][boundary], "segment": original["segment"][boundary]}
    boundary_generated = np.isin(generated["segment"], [0, 1])
    after = {"coord": generated["coord"][boundary_generated], "segment": generated["segment"][boundary_generated]}
    fig = plt.figure(figsize=(12, 5))
    created = np.asarray(generated.get("generated_point", np.zeros(len(generated["coord"]), dtype=bool)), dtype=bool)
    highlight = generated["coord"][boundary_generated & created]
    for i, (scene, title) in enumerate(((before, "Observed wall / floor"), (after, "Completed wall / floor (new points black)"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _scatter(ax, scene, title, int(config.get("seed", 0)), highlight=highlight if i else None)
    fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)
    _write(output.with_suffix(".sources.json"), {"original_scene": matches[0]["id"], "generated_scene": generated_row["id"], "original_boundary_points": len(before["coord"]), "completed_boundary_points": len(after["coord"])})


DIAGRAMS = {
    "figure1": 'flowchart LR\n  A[Training scenes] --> B[Local transforms and scene mixing]\n  A --> C[Extract objects and backgrounds]\n  C --> D[Sample directed relationship graph]\n  D --> E[Local geometry and semantic losses]\n  D --> F[Global graph alignment]\n  E --> G[Optimized new scene]\n  F --> G',
    "figure2": 'flowchart LR\n  A[Training manifest] --> B[Decompose objects/backgrounds]\n  B --> C[Training-only statistics]\n  C --> D[Gaussian class counts + JS selection]\n  D --> E[Sample directional relationship edges]\n  B --> F[Initialize object poses on support surfaces]\n  E --> G[Local losses + frozen pretrained graph encoder GGCL]\n  F --> G\n  G --> H[Optimize xyz/yaw/pitch]\n  H --> I[Original + generated training set]\n  I --> J[Segmentation training and evaluation]',
    "figure3": 'flowchart TB\n  A[Directed pair A to B] --> B[Supported by: xy overlap / base-to-top gap]\n  A --> C[Attached to: overlap / principal-axis alignment]\n  A --> D[Left/right of: oriented half-space overlap]\n  A --> E[Nearby: box-to-box distance]\n  A --> F[Faces: forward direction toward B]\n  A --> G[Oriented with: overlap / parallel normals]\n  A --> H[None: no named rule matches]'
}


def report(config):
    if not config.get("figure"):
        return table_report(config)
    figure = config["figure"]
    output_dir = Path(config.get("output", "artifacts/figures"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if figure in DIAGRAMS:
        path = output_dir / (figure + ".md")
        path.write_text("# " + figure + " implementation diagram\n\n```mermaid\n" + DIAGRAMS[figure] + "\n```\n", encoding="utf-8")
    else:
        path = output_dir / (figure + ".png")
        functions = {"figure4": _scene_comparison, "figure5": _tsne, "figure6": _scene_comparison,
                     "figure7": _scene_comparison, "figure8": _none_instance, "figure9": _completion,
                     "figure10": _class_ablation, "figure11": _class_ablation}
        if figure not in functions:
            raise ValueError("Unknown figure: " + figure)
        if not config.get("sources"):
            raise ValueError("Figure sources are required; obtain a complete config with experiments plan " + figure)
        functions[figure](config, path)
    _write(output_dir / (figure + ".config.json"), config)
    return {"figure": figure, "output": str(path), "source": "computed artifacts"}
