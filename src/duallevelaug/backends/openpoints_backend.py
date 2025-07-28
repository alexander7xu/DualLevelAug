from pathlib import Path
import json
import math
try:
    from .common import (select_scenes, load_scene, checkpoint_path, load_checkpoint,
                        PredictionWriter, file_hash, write_json, safe_id, checkpoint_selection)
except ImportError:  # Direct execution by worker.py.
    from common import (select_scenes, load_scene, checkpoint_path, load_checkpoint,
                        PredictionWriter, file_hash, write_json, safe_id, checkpoint_selection)


def _native_config(cfg, root):
    from openpoints.utils import EasyConfig
    dataset = cfg['dataset'].lower()
    filename = 'pointnet++.yaml' if cfg['model'] == 'pointnet2' else 'pointnext-xl.yaml'
    native = EasyConfig()
    native.load(str(root / 'cfgs' / dataset / filename), recursive=True)
    classes, _ = select_scenes(cfg, 'features' if cfg.get('_action') == 'features' else 'train')
    native.model.cls_args.num_classes = len(classes)
    for key, value in cfg.get('training', {}).items():
        native[key] = value
    for key in ('epochs', 'batch_size', 'lr', 'use_amp'):
        if key in cfg:
            native[key] = cfg[key]
    return native


def _transform_data(coord, color, target, pipeline, keys):
    import numpy as np
    import torch
    data = {'pos': coord.astype(np.float32), 'x': color.astype(np.float32)}
    if target is not None:
        data['y'] = target.astype(np.int64)
    if pipeline is not None:
        data = pipeline(data)
    for key, value in list(data.items()):
        if not torch.is_tensor(value):
            data[key] = torch.from_numpy(np.asarray(value))
    if 'heights' in keys and 'heights' not in data:
        data['heights'] = torch.from_numpy(coord[:, 2:3].astype(np.float32))
    return data


def _make_dataset(scenes, native):
    import torch
    from openpoints.dataset.data_util import crop_pc
    from openpoints.transforms import build_transforms_from_cfg
    pipeline = build_transforms_from_cfg('train', native.datatransforms)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(scenes) * int(native.dataset.train.get('loop', 1))

        def __getitem__(self, index):
            scene = load_scene(scenes[index % len(scenes)])
            if 'color' not in scene:
                raise ValueError('PointNeXt/PointNet++ indoor inputs require RGB')
            coord, color, target = crop_pc(
                scene['coord'], scene['color'], scene['segment'], split='train',
                voxel_size=native.dataset.common.voxel_size,
                voxel_max=native.dataset.train.voxel_max,
                downsample=True, variable=False, shuffle=True)
            return _transform_data(coord, color, target, pipeline, native.feature_keys)
    return Dataset()


def _fragments(coord, voxel_size, max_points=None):
    """Yield deterministic voxel samples covering every input point."""
    import numpy as np
    grid = np.floor((coord - coord.min(0)) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    for offset in range(int(counts.max())):
        selected = order[starts + offset % counts]
        if max_points and len(selected) > max_points:
            # Splitting voxel groups bounds memory but changes scene context.
            for begin in range(0, len(selected), max_points):
                yield selected[begin:begin + max_points]
        else:
            yield selected


def _scene_prediction(model, scene, native, cfg, descriptor=False):
    import numpy as np
    import torch
    from openpoints.transforms import build_transforms_from_cfg
    from openpoints.dataset import get_features_by_keys
    raw = load_scene(scene)
    if 'color' not in raw:
        raise ValueError('RGB is required for indoor OpenPoints models')
    phase = 'test' if native.datatransforms.get('test') else 'val'
    pipeline = build_transforms_from_cfg(phase, native.datatransforms)
    scores = np.zeros((len(raw['coord']), native.model.cls_args.num_classes), dtype=np.float64)
    count = np.zeros(len(scores), dtype=np.int64)
    descriptors = []
    limit = cfg.get('eval_max_points')
    with torch.no_grad():
        for indices in _fragments(raw['coord'], native.dataset.common.voxel_size, limit):
            real_length = len(indices)
            # Four SA stages require at least 1024 points.
            if real_length < 1024:
                indices = np.resize(indices, 1024)
            coord = raw['coord'][indices].copy()
            coord -= coord.min(0)
            data = _transform_data(coord, raw['color'][indices], None, pipeline, native.feature_keys)
            data = {key: value.unsqueeze(0).cuda(non_blocking=True) for key, value in data.items()}
            data['x'] = get_features_by_keys(data, native.feature_keys)
            if descriptor:
                _, features = model.encoder.forward_seg_feat(data)
                last = features[-1] if isinstance(features, (list, tuple)) else features
                descriptors.append(last.flatten(start_dim=2).max(dim=2)[0][0].cpu().numpy())
            else:
                logits = model(data).softmax(1)[0].transpose(0, 1).cpu().numpy()[:real_length]
                np.add.at(scores, indices[:real_length], logits)
                np.add.at(count, indices[:real_length], 1)
    if descriptor:
        return np.stack(descriptors).max(axis=0)
    if not (count > 0).all():
        raise RuntimeError('Evaluation did not cover every original point')
    return scores.argmax(1), raw['segment']


def _validate(model, scenes, classes, native, cfg, save=False, output=None):
    import numpy as np
    model.eval()
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    writer = PredictionWriter(output or cfg['output'], classes, cfg) if save else None
    for index, scene in enumerate(scenes):
        pred, target = _scene_prediction(model, scene, native, cfg)
        valid = (target >= 0) & (target < len(classes))
        cm += np.bincount(len(classes) * target[valid] + pred[valid], minlength=len(classes)**2).reshape(cm.shape)
        if writer:
            writer.add(scene, pred, target)
        print('Evaluate {}/{} {}'.format(index + 1, len(scenes), scene['id']), flush=True)
    if writer:
        return writer.finish()
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    return 100.0 * float((cm.diagonal()[union > 0] / union[union > 0]).mean()) if (union > 0).any() else float('-inf')


def _train(model, cfg, native, classes, scenes):
    import torch
    from openpoints.dataset import get_features_by_keys
    from openpoints.optim import build_optimizer_from_cfg
    from openpoints.scheduler import build_scheduler_from_cfg
    selection = checkpoint_selection(cfg)
    validation = select_scenes(cfg, 'validation')[1] if selection == 'best' else None
    dataset = _make_dataset(scenes, native)
    generator = torch.Generator().manual_seed(int(cfg.get('seed', 0)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(native.batch_size), shuffle=True,
        num_workers=int(cfg.get('num_worker', native.dataloader.num_workers)),
        pin_memory=True, drop_last=False, generator=generator)
    optimizer = build_optimizer_from_cfg(model, lr=native.lr, **native.optimizer)
    scheduler = build_scheduler_from_cfg(native, optimizer)
    amp = bool(native.get('use_amp', False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start, best = 1, float('-inf')
    if cfg.get('checkpoint'):
        checkpoint = load_checkpoint(model, checkpoint_path(cfg))
        if cfg.get('resume'):
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            start, best = checkpoint['epoch'] + 1, checkpoint.get('best_mIoU')
            best = float('-inf') if best is None else best
    output = Path(cfg['output'])
    history = []
    smoothing = float(native.criterion_args.get('label_smoothing', 0.0))
    for epoch in range(start, int(native.epochs) + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for batch in loader:
            data = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            target = data['y'].long()
            data['x'] = get_features_by_keys(data, native.feature_keys)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(data)
                valid = target >= 0
                if not valid.any():
                    continue
                flat = logits.transpose(1, 2)[valid].log_softmax(-1)
                truth = target[valid]
                nll = -flat.gather(1, truth[:, None]).mean()
                loss = (1.0 - smoothing) * nll - smoothing * flat.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss at epoch {}'.format(epoch))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if native.get('grad_norm_clip'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(native.grad_norm_clip))
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler lowers its scale when overflow skips the optimizer step.
            if scaler.get_scale() < scale:
                continue
            total_loss += float(loss.detach())
            batches += 1
        if not batches:
            raise RuntimeError('No optimizer updates completed at epoch {}'.format(epoch))
        scheduler.step(epoch)
        score = _validate(model, validation, classes, native, cfg) if validation else None
        improved = score is not None and score > best
        if score is not None:
            best = max(best, score)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'epoch': epoch, 'best_mIoU': best if validation else None, 'model': cfg['model'],
            'training_dataset': cfg['dataset'], 'training_split': 'train',
            'manifest_sha256': file_hash(cfg['manifest']), 'config': cfg}
        torch.save(checkpoint, str(output / 'last.pt'))
        if improved or (selection == 'last' and epoch == int(native.epochs)):
            torch.save(checkpoint, str(output / 'model.pt'))
        history.append({'epoch': epoch, 'loss': total_loss / batches, 'validation_mIoU': score})
        write_json(output / 'history.json', history)
        print('Epoch {} loss {:.5f} validation mIoU {}'.format(epoch, total_loss / batches, score), flush=True)
    if not (output / 'model.pt').is_file():
        raise RuntimeError('Training did not produce the selected checkpoint')
    return {'checkpoint': str(output / 'model.pt'), 'best_validation_mIoU': best if validation else None, 'checkpoint_selection': selection}


def execute(action, cfg, root):
    import numpy as np
    from openpoints.models import build_model_from_cfg
    if int(cfg.get('num_gpus', 1)) != 1:
        raise ValueError('The OpenPoints adapter requires num_gpus=1')
    cfg = dict(cfg, _action=action)
    native = _native_config(cfg, root)
    write_json(Path(cfg['output']) / 'openpoints_config.json', native.dict())
    model = build_model_from_cfg(native.model).cuda()
    if action == 'train':
        classes, scenes = select_scenes(cfg, 'train')
        return _train(model, cfg, native, classes, scenes)
    checkpoint = load_checkpoint(model, checkpoint_path(cfg))
    model.eval()
    if action == 'features':
        if cfg['model'] != 'pointnet2' or cfg['dataset'].lower() != 's3dis':
            raise ValueError('Feature extraction requires PointNet++ trained on S3DIS')
        if checkpoint.get('training_dataset') != 's3dis' or checkpoint.get('training_split') != 'train':
            raise ValueError('Feature checkpoint must record training_dataset=s3dis and training_split=train; use the pretrain entry in experiment plan')
        if checkpoint.get('manifest_sha256') != file_hash(cfg['manifest']):
            raise ValueError('Feature checkpoint training manifest differs from requested original dataset')
        classes, scenes = select_scenes(cfg, 'features')
        features = [_scene_prediction(model, scene, native, cfg, descriptor=True) for scene in scenes]
        output = Path(cfg['output']) / 'features.npz'
        np.savez_compressed(str(output), features=np.stack(features),
                            scene_ids=np.asarray([s['id'] for s in scenes]),
                            sources=np.asarray(['generated' if s.get('generated') else 'original' for s in scenes]))
        return {'features': str(output), 'num_scenes': len(scenes), 'dimension': len(features[0])}
    classes, scenes = select_scenes(cfg, 'evaluate')
    return _validate(model, scenes, classes, native, cfg, save=True)
