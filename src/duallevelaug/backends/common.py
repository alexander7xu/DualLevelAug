import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_id(value):
    text = str(value)
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', text)[:100] + '_' + hashlib.sha256(text.encode()).hexdigest()[:12]


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data.get('classes') or not isinstance(data.get('scenes'), list):
        raise ValueError('Expected manifest with classes and scenes: ' + str(path))
    seen = set()
    for scene in data['scenes']:
        if scene['id'] in seen:
            raise ValueError('Duplicate scene id: ' + scene['id'])
        seen.add(scene['id'])
        scene['path'] = str((path.parent / scene['path']).resolve())
    return data


def area_of(scene):
    value = scene.get('background_area', scene.get('area', ''))
    return str(value).lower().replace('area_', '').replace('area', '').strip()


def select_scenes(cfg, purpose):
    original = read_manifest(cfg['manifest'])
    generated = read_manifest(cfg['generated_manifest']) if cfg.get('generated_manifest') else None
    if generated and generated['classes'] != original['classes']:
        raise ValueError('Generated and original manifests have different class mappings')
    if purpose in ('train', 'features'):
        scenes = [dict(s, generated=False) for s in original['scenes'] if s['split'] == 'train']
        if generated:
            if any(s['split'] != 'train' for s in generated['scenes']):
                raise ValueError('Generated manifest contains a non-training scene')
            train_ids = set(s['id'] for s in scenes)
            for scene in generated['scenes']:
                source = scene.get('background_source_id', scene.get('background_scene'))
                if not source:
                    raise ValueError('Generated scene lacks background_source_id: ' + scene['id'])
                if source not in train_ids:
                    raise ValueError('Generated background is not from original training split: ' + source)
                if original['dataset'].lower() == 's3dis' and area_of(scene) == '5':
                    raise ValueError('Generated S3DIS training data contains held-out Area 5')
            scenes += [dict(s, generated=True) for s in generated['scenes']]
    else:
        split = 'val' if purpose == 'validation' else cfg.get('eval_split', 'val')
        source = 'original' if purpose == 'validation' else cfg.get('eval_source', 'original')
        if source == 'generated':
            if not generated:
                raise ValueError('Generated-only evaluation requires generated_manifest')
            scenes = [dict(s, generated=True) for s in generated['scenes']]
        elif source == 'original':
            scenes = [dict(s, generated=False) for s in original['scenes']
                      if split == 'all' or s['split'] == split]
        else:
            raise ValueError('eval_source must be original or generated')
        if purpose != 'validation' and cfg.get('eval_area') is not None:
            scenes = [s for s in scenes if area_of(s) == str(cfg['eval_area'])]
    if not scenes:
        raise ValueError('No scenes selected for ' + purpose)
    return original['classes'], scenes


def load_scene(scene, require_normal=False):
    import numpy as np
    with np.load(scene['path'], allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    n = len(data['coord'])
    if n == 0:
        raise ValueError('Empty scene: ' + scene['id'])
    data['coord'] = data['coord'].astype(np.float32)
    data['segment'] = data['segment'].reshape(-1).astype(np.int64)
    if require_normal and 'normal' not in data:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError('This backbone needs normals: install Open3D or prepare normal arrays') from exc
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['coord']))
        cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        cloud.orient_normals_towards_camera_location(data['coord'].mean(0))
        data['normal'] = np.asarray(cloud.normals).astype(np.float32)
    for key in ('normal', 'color', 'strength'):
        if key in data:
            data[key] = data[key].astype(np.float32)
    if 'strength' in data:
        data['strength'] = data['strength'].reshape(n, -1)
    if data['segment'].shape != (n,) or not np.isfinite(data['coord']).all():
        raise ValueError('Malformed scene arrays: ' + scene['id'])
    return data


def verify_backend(cfg):
    names = {'ptv3': 'pointcept', 'ptv2': 'pointcept', 'octformer': 'pointcept',
             'minkunet': 'pointcept', 'pointnext-xl': 'pointnext', 'pointnet2': 'pointnext',
             'sphereformer': 'sphereformer'}
    name = names[cfg['model']]
    lock = json.loads((PACKAGE_ROOT / 'resources' / 'backends.json').read_text())['backends'][name]
    settings = cfg.get('backends', {}).get(name, {})
    workspace = Path(cfg.get('workspace', Path.cwd())).resolve()
    source = Path(cfg.get('backend_root', settings.get('root', workspace / 'external' / name)))
    source = (workspace / source).resolve()
    if not (source / '.git').exists():
        raise RuntimeError('Pinned {} source missing at {}. Run scripts/setup_backends.py --fetch {} first.'.format(name, source, name))
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    if revision != lock['commit']:
        raise RuntimeError('{} revision mismatch: expected {}, got {}'.format(name, lock['commit'], revision))
    status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], universal_newlines=True)
    if status:
        raise RuntimeError('Pinned backend has tracked edits; restore a clean checkout: ' + str(source))
    for child, expected in lock.get('submodules', {}).items():
        actual = subprocess.check_output(['git', '-C', str(source / child), 'rev-parse', 'HEAD'], universal_newlines=True).strip()
        if actual != expected:
            raise RuntimeError('Submodule revision mismatch: ' + child)
    return name, source, lock



def checkpoint_selection(cfg):
    manifest = read_manifest(cfg['manifest'])
    has_validation = any(s['split'] == 'val' for s in manifest['scenes'])
    choice = cfg.get('checkpoint_selection', 'best' if has_validation else 'last')
    if choice not in ('best', 'last'):
        raise ValueError('checkpoint_selection must be best or last')
    if choice == 'best' and not has_validation:
        raise ValueError('Best checkpoint selection requires a validation split; use checkpoint_selection=last')
    return choice


def checkpoint_path(cfg):
    path = cfg.get('checkpoint')
    if not path or not Path(path).is_file():
        raise FileNotFoundError('Evaluation/features require a trained checkpoint in config.checkpoint')
    return str(Path(path).resolve())


def load_checkpoint(model, path):
    import torch
    checkpoint = torch.load(path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PredictionWriter:
    def __init__(self, output, classes, cfg):
        import numpy as np
        self.output, self.classes, self.cfg = Path(output), classes, cfg
        self.directory = self.output / 'predictions'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
        self.records = []

    def add(self, scene, prediction, target):
        import numpy as np
        pred, target = np.asarray(prediction).reshape(-1), np.asarray(target).reshape(-1)
        if pred.shape != target.shape:
            raise ValueError('Prediction/target shape mismatch for ' + scene['id'])
        c = len(self.classes)
        valid = (target >= 0) & (target < c)
        if ((pred[valid] < 0) | (pred[valid] >= c)).any():
            raise ValueError('Out-of-range prediction')
        self.confusion += np.bincount(c * target[valid] + pred[valid], minlength=c*c).reshape(c,c)
        filename = safe_id(scene['id']) + '.npz'
        np.savez_compressed(str(self.directory / filename), pred=pred.astype(np.int32),
                            target=target.astype(np.int32), scene_id=str(scene['id']), area=area_of(scene))
        self.records.append({'id': scene['id'], 'path': str(self.directory / filename),
                             'area': area_of(scene), 'generated': scene.get('generated', False)})

    def finish(self):
        cm = self.confusion
        diag = cm.diagonal()
        union = cm.sum(0) + cm.sum(1) - diag
        iou = [100.0 * int(d) / int(u) if u else None for d, u in zip(diag, union)]
        present = [v for v in iou if v is not None]
        metrics = {'allAcc': 100.0 * int(diag.sum()) / int(cm.sum()) if cm.sum() else None,
                   'mIoU': sum(present) / len(present) if present else None,
                   'IoU': iou, 'class_names': self.classes,
                   'confusion_matrix': cm.tolist(), 'units': 'percent',
                   'num_scenes': len(self.records), 'checkpoint': self.cfg.get('checkpoint'),
                   'checkpoint_sha256': file_hash(self.cfg['checkpoint']) if self.cfg.get('checkpoint') else None}
        write_json(self.output / 'metrics.json', metrics)
        write_json(self.output / 'predictions.json', self.records)
        return metrics
