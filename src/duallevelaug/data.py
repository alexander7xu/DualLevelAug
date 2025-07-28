"""Relative scene paths resolve from the manifest directory."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

SCANNET_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
SCANNET_CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower_curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
S3DIS_CLASSES = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
KITTI_CLASSES = ('car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')
KITTI_LEARNING_MAP = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}
STPLS3D_CLASSES = ('ground', 'building', 'vegetation', 'car', 'light_pole_sign', 'fence')
# Matches the official sequential changeSemLabels implementation (including 8).
STPLS3D_MAP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 1, 8: 3, 9: 1, 10: 1, 11: 4, 12: 4, 13: 1, 14: 5, 15: 0, 16: 0, 17: 1, 18: 0, 19: 0}
DATASET_CLASSES = {'scannet': SCANNET_CLASSES, 's3dis': S3DIS_CLASSES, 'semantickitti': KITTI_CLASSES, 'stpls3d': STPLS3D_CLASSES}


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('Scene preparation requires NumPy; install the core project dependencies described in README.md.') from exc
    return np


def canonical_dataset(name):
    key = name.lower().replace('-', '').replace('_', '')
    if key in ('kitti', 'semkitti'):
        key = 'semantickitti'
    if key not in DATASET_CLASSES:
        raise ValueError('Unknown dataset: %s' % name)
    return key


def map_label(dataset, raw):
    """Training IDs are contiguous; -1 denotes ignored labels."""
    dataset = canonical_dataset(dataset)
    raw = int(raw)
    if dataset == 'scannet':
        return SCANNET_IDS.index(raw) if raw in SCANNET_IDS else -1
    if dataset == 'semantickitti':
        return KITTI_LEARNING_MAP.get(raw, 0) - 1
    if dataset == 'stpls3d':
        return STPLS3D_MAP.get(raw, 0 if raw > 17 else -1)
    return raw if 0 <= raw < len(S3DIS_CLASSES) else -1


def map_labels(dataset, raw):
    np = _numpy()
    raw = np.asarray(raw)
    values, inverse = np.unique(raw, return_inverse=True)
    mapped = np.asarray([map_label(dataset, x) for x in values], dtype=np.int64)
    return mapped[inverse].reshape(raw.shape)


def split_hash(records):
    """Hash logical full-scene membership, independent of machine paths."""
    rows = sorted((str(r['id']), str(r['split']), str(r.get('source_id', r['id']))) for r in records)
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def split_full_scenes(scene_ids, seed=0, fraction=0.8):
    """Split complete scenes before tiling to keep each scene in one split."""
    ids = sorted(set(str(i) for i in scene_ids))
    if not 0 < fraction < 1:
        raise ValueError('Train fraction must be between zero and one')
    if len(ids) < 2:
        raise ValueError('In-domain split requires at least two complete scenes')
    random.Random(seed).shuffle(ids)
    n_train = min(len(ids) - 1, max(1, int(math.floor(len(ids) * fraction))))
    return {identifier: ('train' if i < n_train else 'test') for i, identifier in enumerate(ids)}


def validate_manifest(manifest):
    canonical_dataset(manifest['dataset'])
    scenes = manifest.get('scenes', [])
    ids, membership = set(), {}
    for row in scenes:
        if row['id'] in ids:
            raise ValueError('Duplicate scene ID: %s' % row['id'])
        ids.add(row['id'])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown split: %s' % row['split'])
        source = row.get('source_id', row['id'])
        if source in membership and membership[source] != row['split']:
            raise ValueError('Full source scene leaks between splits: %s' % source)
        membership[source] = row['split']
        if row.get('generated') and row['split'] != 'train':
            raise ValueError('Generated scenes must be marked train')
        if row.get('generated') and row.get('background_split', 'train') != 'train':
            raise ValueError('Generated scene uses held-out background')
    by_id = {row['id']: row for row in scenes}
    for row in scenes:
        if row.get('generated'):
            background_id = row.get('background_scene')
            if background_id in by_id and by_id[background_id]['split'] != 'train':
                raise ValueError('Generated scene references held-out background: %s' % background_id)
    actual = split_hash(scenes)
    if manifest.get('split_hash') and actual != manifest['split_hash']:
        raise ValueError('Manifest split hash does not match its scene records')
    return manifest


def load_manifest(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    for row in manifest['scenes']:
        p = Path(row['path'])
        if not p.is_absolute():
            row['path'] = str((path.parent / p).resolve())
    return manifest


def training_records(manifest):
    """Fit component libraries on original training scenes only."""
    validate_manifest(manifest)
    rows = [row for row in manifest['scenes'] if row['split'] == 'train' and not row.get('generated', False)]
    if not rows:
        raise ValueError('No original training scenes in manifest')
    return rows


def save_scene(path, arrays):
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    if 'coord' not in clean or clean['coord'].ndim != 2 or clean['coord'].shape[1] != 3:
        raise ValueError('coord must have shape [N,3]')
    n = len(clean['coord'])
    if not np.isfinite(clean['coord']).all():
        raise ValueError('Coordinates must be finite')
    for key, value in clean.items():
        if value.dtype.hasobject:
            raise ValueError('Scene NPZ does not support object arrays: %s' % key)
        if key in ('segment', 'instance', 'raw_segment', 'origin', 'color', 'strength', 'normal') and len(value) != n:
            raise ValueError('%s length does not match coordinates' % key)
    clean.setdefault('segment', np.full(n, -1, dtype=np.int64))
    clean.setdefault('instance', np.full(n, -1, dtype=np.int64))
    clean.setdefault('raw_segment', clean['segment'].copy())
    clean.setdefault('origin', np.arange(n, dtype=np.int64))
    clean['coord'] = clean['coord'].astype(np.float32)
    for key in ('segment', 'instance', 'raw_segment', 'origin'):
        clean[key] = clean[key].astype(np.int64)
    with path.open('wb') as f:
        np.savez_compressed(f, **clean)
    return str(path.resolve())


def load_scene(path):
    np = _numpy()
    with np.load(path, allow_pickle=False) as f:
        result = {k: f[k] for k in f.files}
    if 'coord' not in result or 'segment' not in result or 'instance' not in result:
        raise ValueError('Invalid unified scene: %s' % path)
    return result


def _read_text_points(path):
    np = _numpy()
    with Path(path).open(encoding='utf-8-sig') as f:
        first = f.readline()
    delimiter = ',' if ',' in first else None
    try:
        [float(x) for x in (first.strip().split(',') if delimiter else first.split())]
        skiprows = 0
    except ValueError:
        skiprows = 1
    arr = np.loadtxt(str(path), delimiter=delimiter, skiprows=skiprows, ndmin=2)
    if not len(arr) or not np.isfinite(arr).all():
        raise ValueError('Empty or non-finite point file: %s' % path)
    return arr


def read_ply(path):
    """Vertices must be first and have scalar properties; faces are ignored."""
    np = _numpy()
    types = {'char': 'i1', 'uchar': 'u1', 'short': 'i2', 'ushort': 'u2', 'int': 'i4', 'uint': 'u4', 'float': 'f4', 'double': 'f8', 'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2', 'int32': 'i4', 'uint32': 'u4', 'float32': 'f4', 'float64': 'f8'}
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError('Not a PLY file: %s' % path)
        props, count, active, fmt, first_element = [], None, False, None, True
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            tokens = line.decode('ascii').strip().split()
            if not tokens:
                continue
            if tokens[0] == 'end_header':
                break
            if tokens[0] == 'format':
                fmt = tokens[1]
            elif tokens[0] == 'element':
                active = tokens[1] == 'vertex'
                if active:
                    if not first_element:
                        raise ValueError('PLY vertex must be the first element')
                    count = int(tokens[2])
                first_element = False
            elif tokens[0] == 'property' and active:
                if tokens[1] == 'list':
                    raise ValueError('PLY vertex list attributes are unsupported')
                props.append((tokens[2], types[tokens[1]]))
        if count is None:
            raise ValueError('PLY has no vertex element')
        if fmt == 'ascii':
            rows = [f.readline().decode('ascii').split() for _ in range(count)]
            matrix = np.asarray(rows, dtype=np.float64)
            return {name: matrix[:, i].astype(dtype) for i, (name, dtype) in enumerate(props)}
        if fmt not in ('binary_little_endian', 'binary_big_endian'):
            raise ValueError('Unsupported PLY format: %s' % fmt)
        prefix = '<' if fmt == 'binary_little_endian' else '>'
        dtype = np.dtype([(name, prefix + typ) for name, typ in props])
        payload = f.read(dtype.itemsize * count)
        if len(payload) != dtype.itemsize * count:
            raise ValueError('Truncated PLY vertices')
        vertices = np.frombuffer(payload, dtype=dtype, count=count)
        return {name: vertices[name].copy() for name, _ in props}


def _scannet(root, **kwargs):
    np = _numpy()
    split_root = Path(kwargs.get('split_dir') or root)
    split_files = {}
    for split in ('train', 'val', 'test'):
        candidates = [split_root / ('scannetv2_%s.txt' % split), root / 'Tasks' / 'Benchmark' / ('scannetv2_%s.txt' % split), root / 'splits' / ('scannetv2_%s.txt' % split)]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            split_files[split] = found
    if not {'train', 'val'}.issubset(split_files):
        raise FileNotFoundError('ScanNet requires official scannetv2_train.txt and scannetv2_val.txt (root, splits, Tasks/Benchmark or split_dir)')
    tsv = kwargs.get('label_map')
    tsv = Path(tsv) if tsv else next(iter(sorted(root.rglob('*labels.combined.tsv'))), None)
    if tsv is None or not tsv.exists():
        raise FileNotFoundError('ScanNet requires scannetv2-labels.combined.tsv')
    with tsv.open(encoding='utf-8') as f:
        category_map = {r['raw_category']: int(r['nyu40id']) for r in csv.DictReader(f, delimiter='\t')}
    seen = set()
    for split, split_file in split_files.items():
        for scene_id in split_file.read_text().split():
            if scene_id in seen:
                raise ValueError('ScanNet scene occurs in multiple splits: %s' % scene_id)
            seen.add(scene_id)
            candidates = [root / ('scans_test' if split == 'test' else 'scans') / scene_id, root / scene_id]
            folder = next((p for p in candidates if p.is_dir()), None)
            if folder is None:
                raise FileNotFoundError('Missing %s scene from official split: %s' % (split, scene_id))
            ply = folder / (scene_id + '_vh_clean_2.ply')
            vertices = read_ply(ply)
            coord = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float32)
            info = folder / (scene_id + '.txt')
            if info.exists():
                alignment = next((line.split('=', 1)[1] for line in info.read_text().splitlines() if line.startswith('axisAlignment')), None)
                if alignment:
                    matrix = np.asarray([float(x) for x in alignment.split()]).reshape(4, 4)
                    coord = (np.c_[coord, np.ones(len(coord))] @ matrix.T)[:, :3].astype(np.float32)
            raw = np.zeros(len(coord), dtype=np.int64)
            instance = np.full(len(coord), -1, dtype=np.int64)
            if split != 'test':
                seg = json.loads((folder / (scene_id + '_vh_clean_2.0.010000.segs.json')).read_text())
                aggregation_candidates = [folder / (scene_id + '.aggregation.json'), folder / (scene_id + '_vh_clean.aggregation.json')]
                ag_path = next((p for p in aggregation_candidates if p.exists()), aggregation_candidates[0])
                groups = json.loads(ag_path.read_text())['segGroups']
                segids = np.asarray(seg['segIndices'])
                if len(segids) != len(coord):
                    raise ValueError('ScanNet segmentation/vertex length mismatch: %s' % scene_id)
                for group in groups:
                    indices = np.isin(segids, group['segments'])
                    raw[indices] = category_map.get(group['label'], 0)
                    instance[indices] = int(group.get('objectId', group.get('id', 0)))
            arrays = {'coord': coord, 'raw_segment': raw, 'segment': map_labels('scannet', raw), 'instance': instance}
            if all(k in vertices for k in ('red', 'green', 'blue')):
                arrays['color'] = np.column_stack([vertices[k] for k in ('red', 'green', 'blue')]).astype(np.uint8)
            if all(k in vertices for k in ('nx', 'ny', 'nz')):
                arrays['normal'] = np.column_stack([vertices[k] for k in ('nx', 'ny', 'nz')]).astype(np.float32)
                if info.exists() and alignment:
                    arrays['normal'] = (arrays['normal'] @ matrix[:3, :3].T).astype(np.float32)
            yield {'id': scene_id, 'split': split, 'area': None, 'source': str(folder), 'source_id': scene_id}, arrays


def _s3dis(root, **kwargs):
    np = _numpy()
    validation_area = int(kwargs.get('validation_area', 5))
    annotations = sorted(root.rglob('Annotations'))
    if not annotations:
        raise FileNotFoundError('No Area_*/room/Annotations folders found in S3DIS root')
    for folder in annotations:
        area_part = next((p for p in folder.parts if p.startswith('Area_')), None)
        if area_part is None:
            continue
        area = int(area_part.split('_')[1])
        chunks = []
        for i, path in enumerate(sorted(folder.glob('*.txt'))):
            values = _read_text_points(path)
            if values.shape[1] < 6:
                raise ValueError('S3DIS annotations require XYZRGB: %s' % path)
            name = path.stem.rsplit('_', 1)[0]
            # S3DIS contains occasional "stairs" annotations; official converters
            # map unknown annotations to clutter, rather than dropping points.
            label = S3DIS_CLASSES.index(name) if name in S3DIS_CLASSES else 12
            chunks.append((values[:, :6], np.full(len(values), label), np.full(len(values), i)))
        if not chunks:
            raise ValueError('Empty S3DIS annotation folder: %s' % folder)
        points = np.concatenate([x[0] for x in chunks])
        labels = np.concatenate([x[1] for x in chunks]).astype(np.int64)
        identifier = area_part + '_' + folder.parent.name
        yield {'id': identifier, 'split': 'val' if area == validation_area else 'train', 'area': area, 'source': str(folder), 'source_id': identifier}, {'coord': points[:, :3], 'color': np.clip(points[:, 3:6], 0, 255).astype(np.uint8), 'segment': labels, 'raw_segment': labels.copy(), 'instance': np.concatenate([x[2] for x in chunks]).astype(np.int64)}


def _semantickitti(root, **kwargs):
    np = _numpy()
    sequences = root / 'sequences'
    if not sequences.is_dir():
        sequences = root / 'dataset' / 'sequences'
    if not sequences.is_dir():
        raise FileNotFoundError('SemanticKITTI root must contain sequences/ or dataset/sequences/')
    found = False
    for folder in sorted(sequences.iterdir()):
        if not folder.is_dir() or not folder.name.isdigit():
            continue
        seq = int(folder.name)
        if not 0 <= seq <= 21:
            continue
        split = 'val' if seq == 8 else ('train' if seq <= 10 else 'test')
        for path in sorted((folder / 'velodyne').glob('*.bin')):
            found = True
            points = np.fromfile(path, dtype='<f4').reshape(-1, 4)
            labels_path = folder / 'labels' / (path.stem + '.label')
            if split != 'test' and not labels_path.exists():
                raise FileNotFoundError('Missing SemanticKITTI labels: %s' % labels_path)
            packed = np.fromfile(labels_path, dtype='<u4') if labels_path.exists() else np.zeros(len(points), dtype=np.uint32)
            if len(packed) != len(points):
                raise ValueError('SemanticKITTI point/label length mismatch: %s' % path)
            raw = (packed & 0xFFFF).astype(np.int64)
            instance = (packed >> 16).astype(np.int64)
            instance[instance == 0] = -1
            identifier = '%02d_%s' % (seq, path.stem)
            yield {'id': identifier, 'split': split, 'area': '%02d' % seq, 'source': str(path), 'source_id': identifier, 'sequence': '%02d' % seq}, {'coord': points[:, :3], 'strength': points[:, 3:4], 'raw_segment': raw, 'segment': map_labels('semantickitti', raw), 'instance': instance}
    if not found:
        raise FileNotFoundError('No SemanticKITTI velodyne/*.bin files found')


def _stpls3d(root, protocol='default', seed=0, **kwargs):
    np = _numpy()
    protocol = {'default': 'cross-domain', 'cross_domain': 'cross-domain', 'in_domain': 'in-domain', 'synthetic': 'in-domain', 'wmsc': 'cross-domain'}.get(protocol, protocol)
    if protocol not in ('cross-domain', 'in-domain'):
        raise ValueError('STPLS3D protocol must be cross-domain or in-domain')
    files = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.txt', '.ply') and p.is_file())
    synthetic, real = [], []
    for path in files:
        normalized = str(path.relative_to(root)).lower().replace('_', '').replace('-', '')
        if 'syntheticv3' in normalized:
            synthetic.append(path)
        elif 'wmsc' in normalized:
            real.append(path)
    if not synthetic:
        raise FileNotFoundError('STPLS3D needs complete scenes under Synthetic_v3/ (TXT or PLY)')
    ids = {str(p): p.relative_to(root).with_suffix('').as_posix().replace('/', '__') for p in synthetic + real}
    assignment = split_full_scenes([ids[str(p)] for p in synthetic], seed) if protocol == 'in-domain' else {ids[str(p)]: 'train' for p in synthetic}
    if protocol == 'cross-domain' and not real:
        raise FileNotFoundError('Cross-domain STPLS3D requires WMSC raw scene(s)')
    selected = synthetic + (real if protocol == 'cross-domain' else [])
    for path in selected:
        identifier = ids[str(path)]
        if path.suffix.lower() == '.ply':
            v = read_ply(path)
            coord = np.column_stack([v[k] for k in ('x', 'y', 'z')])
            color = np.column_stack([v[k] for k in ('red', 'green', 'blue')])
            label_key = next((k for k in ('class', 'semantic', 'label') if k in v), None)
            if label_key is None:
                raise ValueError('STPLS3D PLY missing semantic class')
            raw = v[label_key].astype(np.int64)
            instance = v.get('instance', v.get('instance_id', np.full(len(coord), -1))).astype(np.int64)
        else:
            a = _read_text_points(path)
            if a.shape[1] < 7:
                raise ValueError('STPLS3D requires XYZRGB semantic [instance]: %s' % path)
            coord, color, raw = a[:, :3], a[:, 3:6], a[:, 6].astype(np.int64)
            instance = a[:, 7].astype(np.int64) if a.shape[1] >= 8 else np.full(len(a), -1, dtype=np.int64)
        instance[instance < 0] = -1
        # Keep a local metric frame to avoid float32 UTM precision loss; record offset.
        offset = coord.min(axis=0).astype(np.float64)
        yield {'id': identifier, 'split': assignment.get(identifier, 'test'), 'area': 'Synthetic_v3' if path in synthetic else 'WMSC', 'source': str(path), 'source_id': identifier, 'coordinate_offset': offset.tolist()}, {'coord': coord - offset, 'color': np.clip(color, 0, 255).astype(np.uint8), 'raw_segment': raw, 'segment': map_labels('stpls3d', raw), 'instance': instance}


def prepare(dataset, root, output, protocol='default', seed=0, **kwargs):
    dataset = canonical_dataset(dataset)
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError('Dataset root does not exist: %s' % root)
    converter = {'scannet': _scannet, 's3dis': _s3dis, 'semantickitti': _semantickitti, 'stpls3d': _stpls3d}[dataset]
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for record, arrays in converter(root, protocol=protocol, seed=seed, **kwargs):
        record['path'] = save_scene(output / 'scenes' / (record['id'] + '.npz'), arrays)
        record['num_points'] = len(arrays['coord'])
        records.append(record)
    if not records:
        raise ValueError('No scenes were prepared')
    manifest = {'schema_version': 1, 'dataset': dataset, 'classes': list(DATASET_CLASSES[dataset]), 'ignore_index': -1, 'units': 'meters', 'source_root': str(root), 'protocol': protocol, 'seed': seed, 'scenes': records, 'split_hash': split_hash(records), 'preparation': kwargs}
    validate_manifest(manifest)
    destination = output / 'manifest.json'
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return str(destination)
