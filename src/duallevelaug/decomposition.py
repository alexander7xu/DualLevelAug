from __future__ import annotations

import json
import math
from pathlib import Path

from .data import _numpy, canonical_dataset, load_manifest, load_scene, save_scene, training_records

KITTI_FOREGROUND_RAW = frozenset((10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259))
KITTI_FLOOR_RAW = frozenset((40, 44, 48, 49, 60))


def category_roles(dataset, scene):
    """SemanticKITTI roles use raw IDs before learning-map merging."""
    np = _numpy()
    dataset = canonical_dataset(dataset)
    labels = np.asarray(scene['segment'])
    if dataset == 'semantickitti':
        if 'raw_segment' not in scene:
            raise ValueError('SemanticKITTI decomposition requires raw_segment, before learning-map merging')
        raw = np.asarray(scene['raw_segment'])
        floor = np.isin(raw, list(KITTI_FLOOR_RAW))
        foreground = np.isin(raw, list(KITTI_FOREGROUND_RAW))
    elif dataset == 'scannet':
        floor = labels == 1
        foreground = (labels >= 2)
    elif dataset == 's3dis':
        floor = labels == 1
        foreground = (labels >= 0) & ~np.isin(labels, (1, 2))
    else:
        # STPLS3D has no published decomposition; vehicles/bikes are movable.
        # Buildings, vegetation, poles and fences remain static.
        raw = np.asarray(scene.get('raw_segment', labels))
        floor = labels == 0
        foreground = np.isin(raw, (5, 6, 7, 8, 9, 10))
    background = ~(floor | foreground)
    return {'floor': floor, 'background': background, 'foreground': foreground}


def pca_frame(coord):
    """Right-handed frame with world Z up; PCA does not resolve front/back.

    The largest XY loading is positive; degenerate data use an upright frame.
    """
    np = _numpy()
    points = np.asarray(coord, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError('PCA requires nonempty [N,3] coordinates')
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    eigenvalues = values[::-1].clip(0)
    xy = centered[:, :2].T @ centered[:, :2] / max(1, len(points) - 1)
    xy_values, xy_axes = np.linalg.eigh(xy)
    front = np.array([1., 0., 0.])
    if xy_values[-1] > 1e-12 and xy_values[-1] - xy_values[0] > xy_values[-1] * 1e-6:
        front[:2] = xy_axes[:, -1]
        if front[np.argmax(np.abs(front[:2]))] < 0:
            front *= -1
    up = np.array([0., 0., 1.])
    side = np.cross(up, front)
    rotation = np.column_stack((front, side, up))
    local = centered @ rotation
    normal = vectors[:, 0]
    # For nonplanar instances, the smallest PCA axis is not a support normal.
    planar = bool(eigenvalues[1] > 1e-12 and eigenvalues[2] / eigenvalues[1] < .1)
    if not planar:
        normal = up.copy()
    if normal[2] < 0 or (abs(normal[2]) < 1e-10 and normal[np.argmax(np.abs(normal))] < 0):
        normal = -normal
    return {'centroid': center, 'rotation': rotation, 'extent': np.ptp(local, axis=0), 'front': front, 'normal': normal, 'eigenvalues': eigenvalues, 'pca_degenerate': bool(eigenvalues[1] < 1e-12), 'pca_front_ambiguous': True}


def connected_components(coord, radius=0.35):
    """Query neighbors per point to avoid an N x N distance matrix."""
    np = _numpy()
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('Instance clustering requires SciPy; install the core project dependencies described in README.md.') from exc
    points = np.asarray(coord)
    if radius <= 0:
        raise ValueError('Component radius must be positive')
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    for start in range(len(points)):
        if visited[start]:
            continue
        visited[start] = True
        stack, members = [start], []
        while stack:
            index = stack.pop()
            members.append(index)
            neighbors = np.asarray(tree.query_ball_point(points[index], radius), dtype=np.int64)
            unseen = neighbors[~visited[neighbors]]
            visited[unseen] = True
            stack.extend(unseen.tolist())
        yield np.asarray(members, dtype=np.int64)


def subset_scene(scene, indices):
    np = _numpy()
    n = len(scene['coord'])
    return {k: np.asarray(v)[indices].copy() for k, v in scene.items() if hasattr(v, 'shape') and v.ndim > 0 and len(v) == n}


def _component(scene, indices, class_id, instance_id, role):
    result = subset_scene(scene, indices)
    frame = pca_frame(result['coord'])
    frame['normal_axis'] = frame.pop('normal')
    result.update(frame)
    result.update({'is_floor': role == 'floor', 'class_id': int(class_id), 'instance_id': int(instance_id), 'source_indices': indices, 'role': role})
    return result


def detect_support_planes(coord, distance_threshold=.03, iterations=300, min_points=30, max_planes=16, seed=0, max_fit_points=20000, normal_angle=20.):
    """Unit normals satisfy n.x + offset = 0; inliers index the original input.

    Fit sampled points with RANSAC/PCA, then validate against all remaining points.
    """
    np = _numpy()
    points = np.asarray(coord, dtype=np.float64)
    if len(points) < 3:
        return []
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points))
    planes = []
    cos_angle = math.cos(math.radians(normal_angle))
    for _ in range(max_planes):
        if len(remaining) < max(3, min_points):
            break
        fit_indices = rng.choice(remaining, max_fit_points, replace=False) if len(remaining) > max_fit_points else remaining
        sample = points[fit_indices]
        best_mask, best_count = None, 0
        for _trial in range(iterations):
            a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            normal /= length
            if normal_angle < 90 and abs(normal[2]) < cos_angle:
                continue
            mask = np.abs((sample - a) @ normal) <= distance_threshold
            count = int(mask.sum())
            if count > best_count:
                best_mask, best_count = mask, count
        if best_mask is None or best_count < max(3, min_points * len(sample) / len(remaining)):
            break
        fitted = sample[best_mask]
        center = fitted.mean(axis=0)
        _values, vectors = np.linalg.eigh((fitted - center).T @ (fitted - center))
        normal = vectors[:, 0]
        if normal[2] < -1e-10 or (abs(normal[2]) < 1e-10 and normal[np.argmax(np.abs(normal))] < 0):
            normal = -normal
        offset = -float(center @ normal)
        inlier_mask = np.abs(points[remaining] @ normal + offset) <= distance_threshold
        inliers = remaining[inlier_mask]
        if len(inliers) < min_points:
            break
        surface = points[inliers]
        plane = {'normal': normal, 'offset': offset, 'center': surface.mean(axis=0), 'bounds': np.stack((surface.min(axis=0), surface.max(axis=0))), 'indices': inliers}
        existing = next((p for p in planes if float(p['normal'] @ normal) >= math.cos(math.radians(10)) and abs(p['offset'] - offset) <= 2 * distance_threshold), None)
        if existing is None:
            planes.append(plane)
        else:
            existing['indices'] = np.concatenate((existing['indices'], inliers))
            merged = points[existing['indices']]
            existing['center'] = merged.mean(axis=0)
            existing['bounds'] = np.stack((merged.min(axis=0), merged.max(axis=0)))
        remaining = remaining[~inlier_mask]
    return planes


def _static_groups(scene, indices, role, min_points, seed, radius):
    """Plane partitioning prevents a room's walls sharing one enclosing box."""
    np = _numpy()
    if role == 'floor' or len(indices) < max(30, min_points):
        return [indices]
    points = scene['coord'][indices]
    planes = detect_support_planes(points, min_points=max(30, min_points),
                                   max_planes=32, seed=seed, normal_angle=90.)
    used = np.zeros(len(indices), dtype=bool)
    groups = []
    for plane in planes:
        local = plane['indices']
        used[local] = True
        groups.append(indices[local])
    remaining = indices[~used]
    if len(remaining):
        for part in connected_components(scene['coord'][remaining], radius=radius):
            groups.append(remaining[part])
    return groups


def decompose(scene, dataset, min_points=20, component_radius=.35, seed=0, **kwargs):
    """Components retain world coordinates; instance IDs are scoped to class IDs.
    Small or unresolved objects remain in the fixed background.
    """
    np = _numpy()
    roles = category_roles(dataset, scene)
    labels, instances = np.asarray(scene['segment']), np.asarray(scene['instance'])
    raw = np.asarray(scene.get('raw_segment', labels))
    objects, boundaries = [], []
    moved = np.zeros(len(labels), dtype=bool)
    next_instance = int(max(-1, instances.max(initial=-1))) + 1
    for role in ('foreground', 'floor', 'background'):
        # Raw IDs keep distinct outdoor classes from merging into one object.
        grouping_labels = raw if canonical_dataset(dataset) in ('semantickitti', 'stpls3d') else labels
        for semantic in np.unique(grouping_labels[roles[role]]):
            semantic_indices = np.flatnonzero(roles[role] & (grouping_labels == semantic))
            for instance_id in np.unique(instances[semantic_indices]):
                indices = semantic_indices[instances[semantic_indices] == instance_id]
                groups = [indices] if role == 'foreground' else _static_groups(scene, indices, role, min_points, seed, component_radius)
                if instance_id < 0 and role == 'foreground':
                    groups = [indices[part] for part in connected_components(scene['coord'][indices], component_radius)]
                for group in groups:
                    if len(group) < min_points and role == 'foreground':
                        if int(labels[group[0]]) >= 0:
                            fixed = _component(scene, group, int(labels[group[0]]), int(instance_id), 'background')
                            fixed['raw_class_id'] = int(semantic)
                            fixed['retained_foreground'] = True
                            boundaries.append(fixed)
                        continue
                    class_id = int(labels[group[0]])
                    if class_id < 0:
                        continue
                    identifier = int(instance_id)
                    if identifier < 0:
                        identifier = next_instance
                        next_instance += 1
                    component = _component(scene, group, class_id, identifier, role)
                    component['raw_class_id'] = int(semantic)
                    if role == 'foreground':
                        objects.append(component)
                        moved[group] = True
                    else:
                        boundaries.append(component)
    fixed_indices = np.flatnonzero(~moved)
    background = subset_scene(scene, fixed_indices)
    background['instance'][:] = -1
    floor_indices = np.flatnonzero(roles['floor'])
    supports = detect_support_planes(scene['coord'][floor_indices], seed=seed)
    for plane in supports:
        plane['source_indices'] = floor_indices[plane['indices']]
    if not supports:
        supports = detect_support_planes(background['coord'], seed=seed)
        for plane in supports:
            plane['source_indices'] = fixed_indices[plane['indices']]
    return {'objects': objects, 'background': background, 'boundaries': boundaries, 'background_components': boundaries, 'supports': supports, 'scene_bounds': np.stack((scene['coord'].min(axis=0), scene['coord'].max(axis=0))), 'fixed_indices': fixed_indices, 'counts': {'foreground': len(objects), 'floor': sum(c['role'] == 'floor' for c in boundaries), 'background': sum(c['role'] == 'background' for c in boundaries)}}


def _open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError('Poisson completion requires Open3D; install the project geometry extra as described in README.md.') from exc
    return o3d


def inherit_background_attributes(background, new_coord, class_id, seed=0, noise_std=.002):
    """Use the nearest same-class source; origin keeps its original point index."""
    np = _numpy()
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('Poisson attribute propagation requires SciPy') from exc
    candidate = np.flatnonzero(np.asarray(background['segment']) == class_id)
    if not len(candidate):
        raise ValueError('No same-class background points for label %s' % class_id)
    coordinates = np.asarray(new_coord, dtype=np.float64)
    _distance, nearest = cKDTree(background['coord'][candidate]).query(coordinates)
    source = candidate[nearest]
    result = subset_scene(background, source)
    result['coord'] = (coordinates + np.random.default_rng(seed).normal(0, noise_std, coordinates.shape)).astype(np.float32)
    result['segment'][:] = int(class_id)
    result['instance'][:] = -1
    result['generated_point'] = np.ones(len(coordinates), dtype=np.uint8)
    result['origin'] = np.asarray(background.get('origin', np.arange(len(background['coord']))))[source].astype(np.int64)
    return result


def complete_background(scene, background, depth=9, distance_threshold=.10, normal_angle=20., density_trim=.01, noise_std=.002, seed=0, min_spacing=.01, max_new_ratio=1.0, class_ids=(0, 1)):
    """Classwise Poisson completion, bounded by class extents plus distance_threshold.

    Defaults for parameters unspecified in the paper are implementation choices.
    """
    np = _numpy()
    o3d = _open3d()
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError('Poisson completion requires SciPy') from exc
    if not 0 <= density_trim < 1 or depth < 2 or max_new_ratio < 0:
        raise ValueError('Invalid Poisson completion parameters')
    if len(background['coord']) < 10:
        return {k: v.copy() for k, v in background.items()}
    rng = np.random.default_rng(seed)
    o3d.utility.random.seed(seed)
    raw_cloud = o3d.geometry.PointCloud()
    raw_cloud.points = o3d.utility.Vector3dVector(np.asarray(scene['coord'], dtype=np.float64))
    raw_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(.1, distance_threshold * 2), max_nn=30))
    raw_normals = np.asarray(raw_cloud.normals)
    output = {k: v.copy() for k, v in background.items()}
    output.setdefault('origin', np.arange(len(output['coord']), dtype=np.int64))
    output.setdefault('generated_point', np.zeros(len(output['coord']), dtype=np.uint8))
    additions = []
    for class_id in np.unique(background['segment']):
        if class_id < 0 or (class_ids is not None and class_id not in class_ids):
            continue
        mask = np.asarray(background['segment']) == class_id
        gt = np.asarray(background['coord'])[mask]
        if len(gt) < 10:
            continue
        gt_cloud = o3d.geometry.PointCloud()
        gt_cloud.points = o3d.utility.Vector3dVector(gt.astype(np.float64))
        gt_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(.1, distance_threshold * 2), max_nn=30))
        gt_normals = np.asarray(gt_cloud.normals)
        distances, nearest = cKDTree(gt).query(scene['coord'])
        # Normal signs are ambiguous; compare planes using the absolute dot product.
        compatible = np.abs((raw_normals * gt_normals[nearest]).sum(axis=1)) >= math.cos(math.radians(normal_angle))
        selected = (distances <= distance_threshold) & compatible
        coarse_coord = np.concatenate((gt, np.asarray(scene['coord'])[selected]), axis=0)
        coarse_normal = np.concatenate((gt_normals, raw_normals[selected]), axis=0)
        _d, near_gt = cKDTree(gt).query(coarse_coord)
        signs = np.where((coarse_normal * gt_normals[near_gt]).sum(axis=1) < 0, -1., 1.)
        coarse_normal *= signs[:, None]
        coarse = o3d.geometry.PointCloud()
        coarse.points = o3d.utility.Vector3dVector(coarse_coord)
        coarse.normals = o3d.utility.Vector3dVector(coarse_normal)
        try:
            coarse.orient_normals_consistent_tangent_plane(min(30, len(coarse_coord) - 1))
        except RuntimeError:
            # Coplanar inputs make QHull singular; reuse aligned normals.
            reference = gt_normals[0]
            coarse_normal *= np.where(coarse_normal @ reference < 0, -1., 1.)[:, None]
            coarse.normals = o3d.utility.Vector3dVector(coarse_normal)
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(coarse, depth=depth)
        vertices, density = np.asarray(mesh.vertices), np.asarray(density)
        if not len(vertices):
            raise RuntimeError('Poisson reconstruction produced an empty mesh for class %d' % class_id)
        keep = density >= np.quantile(density, density_trim)
        keep &= ((vertices >= gt.min(axis=0) - distance_threshold) & (vertices <= gt.max(axis=0) + distance_threshold)).all(axis=1)
        vertices = vertices[keep]
        if not len(vertices):
            continue
        distance, _nearest = cKDTree(gt).query(vertices)
        vertices = vertices[distance >= min_spacing]
        maximum = int(math.ceil(len(gt) * max_new_ratio))
        if len(vertices) > maximum:
            vertices = vertices[rng.choice(len(vertices), maximum, replace=False)]
        if len(vertices):
            additions.append(inherit_background_attributes(output, vertices, int(class_id), seed=seed + int(class_id), noise_std=noise_std))
    for addition in additions:
        for key in output:
            if key in addition:
                output[key] = np.concatenate((output[key], addition[key]), axis=0)
    return output


def build_library(manifest_path, output, poisson=False, seed=0, **kwargs):
    np = _numpy()
    manifest = load_manifest(manifest_path)
    rows = training_records(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    library = {'schema_version': 1, 'dataset': manifest['dataset'], 'classes': manifest['classes'], 'split_hash': manifest['split_hash'], 'seed': seed, 'poisson': poisson, 'objects': [], 'backgrounds': [], 'counts': {'foreground': 0, 'floor': 0, 'background': 0}}
    for i, row in enumerate(rows):
        scene = load_scene(row['path'])
        parts = decompose(scene, manifest['dataset'], seed=seed + i, **kwargs)
        background = parts['background']
        if poisson:
            completion_classes = {'scannet': (0, 1), 's3dis': (1, 2), 'semantickitti': (8, 9, 10, 11, 12), 'stpls3d': (0, 1)}[manifest['dataset']]
            background = complete_background(scene, background, seed=seed + i, class_ids=completion_classes)
        background_path = save_scene(output / 'backgrounds' / (row['id'] + '.npz'), background)
        # Keep NPZ attributes point-aligned; store geometry in JSON.
        supports = [{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in p.items() if k != 'indices'} for p in parts['supports']]
        library['backgrounds'].append({'id': row['id'], 'path': background_path, 'split': 'train', 'source_id': row['id'], 'area': row.get('area'), 'source': row['source'], 'supports': supports, 'scene_bounds': parts['scene_bounds'].tolist()})
        for j, obj in enumerate(parts['objects']):
            identifier = '%s__%05d' % (row['id'], j)
            arrays = {k: v for k, v in obj.items() if k in scene and hasattr(v, 'ndim') and v.ndim > 0 and len(v) == len(obj['coord'])}
            path = save_scene(output / 'objects' / (identifier + '.npz'), arrays)
            library['objects'].append({'id': identifier, 'path': path, 'split': 'train', 'source_id': row['id'], 'class_id': obj['class_id'], 'raw_class_id': obj['raw_class_id'], 'instance_id': obj['instance_id'], 'centroid': obj['centroid'].tolist(), 'rotation': obj['rotation'].tolist(), 'extent': obj['extent'].tolist(), 'front': obj['front'].tolist(), 'normal_axis': obj['normal_axis'].tolist(), 'pca_degenerate': obj['pca_degenerate']})
        for key in library['counts']:
            library['counts'][key] += parts['counts'][key]
    destination = output / 'library.json'
    destination.write_text(json.dumps(library, indent=2) + '\n', encoding='utf-8')
    return str(destination)


poisson_complete = complete_background
