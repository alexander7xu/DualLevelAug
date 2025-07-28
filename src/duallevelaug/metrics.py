from pathlib import Path
from .utils import require, write_json


def confusion_matrix(pred, target, num_classes):
    np = require("numpy")
    pred, target = np.asarray(pred).reshape(-1), np.asarray(target).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError("Prediction and target lengths differ")
    valid = (target >= 0) & (target < num_classes)
    if np.any((pred[valid] < 0) | (pred[valid] >= num_classes)):
        raise ValueError("Prediction outside class range on a labeled point")
    return np.bincount(num_classes * target[valid].astype(int) + pred[valid].astype(int),
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_confusion(cm):
    np = require("numpy")
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or np.any(cm < 0):
        raise ValueError("Expected nonnegative square confusion matrix")
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    valid = union > 0
    iou = np.divide(cm.diagonal(), union, out=np.zeros_like(union), where=valid)
    return {"allAcc": float(100 * cm.trace() / cm.sum()) if cm.sum() else None,
            "mIoU": float(100 * iou[valid].mean()) if valid.any() else None,
            "IoU": [float(x * 100) if ok else None for x, ok in zip(iou, valid)],
            "confusion": cm.astype(int).tolist(), "unit": "percent",
            "absent_class_policy": "exclude classes with zero union"}


def evaluate_predictions(pred_dir, output, num_classes):
    np = require("numpy")
    paths = sorted(Path(pred_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError("No prediction NPZ files in %s" % pred_dir)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_scene = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            scene_cm = confusion_matrix(a["pred"], a["target"], num_classes)
        cm += scene_cm
        per_scene[path.stem] = metrics_from_confusion(scene_cm)
    result = metrics_from_confusion(cm)
    result["scenes"] = per_scene
    write_json(output, result)
    return result
