from copy import deepcopy
from .utils import read_json

DEFAULTS = {
    "seed": 0, "ratio": 0.25, "constraints": "full", "poisson": True,
    "gt_sampling": False, "candidates": 64, "edge_noise": 0.05,
    "cooccurrence_threshold": 0.0, "steps": 300, "pose_lr": 0.03,
    "collision_thickness": 0.02, "layout_attempts": 64,
    "loss_weights": {"geometric": 1.0, "semantic": 1.0, "topology": 1.0},
    "relation_thresholds": {"overlap": 0.2, "support_gap": 0.10, "attach": 0.1,
                            "direction": 0.95, "left": 0.8, "right": 0.8,
                            "near": 1.0, "face": 0.8, "normal": 0.95,
                            "temperature": 0.1},
    "poisson_options": {"depth": 9, "distance": 0.10, "normal_angle": 20,
                        "density_trim": 0.01, "jitter": 0.002},
    "graph": {"hidden": 128, "layers": 3, "mask_ratio": 0.2,
              "epochs": 100, "lr": 0.001},
}


def merge(base, override):
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(path):
    raw = read_json(path)
    cfg = merge(DEFAULTS, raw)
    if cfg.get("dataset") == "scannet" and "gt_sampling" not in raw:
        cfg["gt_sampling"] = True
    if "poisson" not in raw and cfg.get("dataset") not in ("scannet","s3dis"):
        cfg["poisson"] = False
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg.get("dataset") not in (None, "scannet", "s3dis", "semantic_kitti", "semantickitti", "stpls3d"):
        raise ValueError("Unknown dataset")
    if cfg.get("constraints", "full") not in ("random", "geometric", "local", "full"):
        raise ValueError("constraints must be random/geometric/local/full")
    if not 0 <= cfg.get("ratio", 0.25) <= 10:
        raise ValueError("ratio must be between 0 and 10")
    if cfg.get("candidates", 64) < 1 or cfg.get("steps", 300) < 0:
        raise ValueError("candidates must be positive and steps nonnegative")
    return cfg
