from pathlib import Path
try:
    from .common import (select_scenes, load_scene, checkpoint_path, load_checkpoint,
                        PredictionWriter, file_hash, write_json, checkpoint_selection)
except ImportError:  # Direct execution by worker.py.
    from common import (select_scenes, load_scene, checkpoint_path, load_checkpoint,
                        PredictionWriter, file_hash, write_json, checkpoint_selection)


def _configuration(cfg, root):
    import yaml
    raw = yaml.safe_load((root / 'config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml').read_text())
    native = {}
    for section in raw.values():
        native.update(section)
    native.update(cfg.get('training', {}))
    for key in ('epochs', 'batch_size', 'use_amp'):
        if key in cfg:
            native[key] = cfg[key]
    return native


def _model(native):
    import numpy as np
    from model.unet_spherical_transformer import Semantic
    window = np.asarray(native['voxel_size'], dtype=np.float32) * native['patch_size'] * native['window_size']
    sphere_window = np.asarray(native['window_size_sphere'], dtype=np.float32)
    return Semantic(input_c=native['input_c'], m=native['m'], classes=native['classes'],
        block_reps=native['block_reps'], block_residual=native['block_residual'], layers=native['layers'],
        window_size=window, window_size_sphere=sphere_window,
        quant_size=window / native['quant_size_scale'], quant_size_sphere=sphere_window / native['quant_size_scale'],
        rel_query=native['rel_query'], rel_key=native['rel_key'], rel_value=native['rel_value'],
        drop_path_rate=native['drop_path_rate'], window_size_scale=native['window_size_scale'],
        grad_checkpoint_layers=native['grad_checkpoint_layers'], sphere_layers=native['sphere_layers'], a=native['a']).cuda()


def _prepare(scene, native, training):
    import numpy as np
    from util.data_util import data_prepare
    raw = load_scene(scene)
    if 'strength' not in raw:
        raise ValueError('SphereFormer needs SemanticKITTI reflectance strength')
    points = np.concatenate([raw['coord'], raw['strength'][:, :1]], axis=1)
    if training:
        angle = np.random.uniform(-np.pi, np.pi)
        cosine, sine = np.cos(angle), np.sin(angle)
        points[:, :2] = points[:, :2].dot(np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float32))
        flip = np.random.randint(4)
        if flip in (1, 3):
            points[:, 0] *= -1
        if flip in (2, 3):
            points[:, 1] *= -1
        points[:, :2] *= np.random.uniform(0.95, 1.05)
        points[:, :3] += np.random.normal(0.0, 0.1, (1, 3))
    xyz = points[:, :3]
    if native.get('pc_range'):
        xyz = np.clip(xyz, native['pc_range'][0], native['pc_range'][1])
    labels = raw['segment'].copy()
    labels[labels < 0] = int(native['ignore_label'])
    return data_prepare(xyz, points, labels, 'train' if training else 'val',
                        np.asarray(native['voxel_size'], dtype=np.float32),
                        native.get('voxel_max') if training else None,
                        None, native.get('xyz_norm', False))


def _batch(samples):
    import torch
    coordinates, xyz, features, labels, batches = [], [], [], [], []
    for index, sample in enumerate(samples):
        coord, pos, feat, label = sample[:4]
        batch = torch.full((len(coord),), index, dtype=torch.long)
        coordinates.append(torch.cat([batch[:, None], coord], dim=1))
        xyz.append(pos)
        features.append(feat)
        labels.append(label)
        batches.append(batch)
    return (torch.cat(coordinates), torch.cat(xyz), torch.cat(features),
            torch.cat(labels), torch.cat(batches), len(samples))


def _forward(model, packed):
    import numpy as np
    import spconv.pytorch as spconv
    coord, xyz, features, labels, batch, size = packed
    spatial_shape = np.clip(coord[:, 1:].max(0)[0].numpy() + 1, 128, None)
    sparse = spconv.SparseConvTensor(features.cuda(non_blocking=True), coord.int().cuda(non_blocking=True), spatial_shape, size)
    logits = model(sparse, xyz.cuda(non_blocking=True), batch.cuda(non_blocking=True))
    return logits, labels.cuda(non_blocking=True)


def _evaluate(model, scenes, classes, native, cfg, save=False):
    import numpy as np
    import torch
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    writer = PredictionWriter(cfg['output'], classes, cfg) if save else None
    model.eval()
    with torch.no_grad():
        for index, scene in enumerate(scenes):
            sample = _prepare(scene, native, False)
            logits, target = _forward(model, _batch([sample]))
            pred = logits[sample[4].cuda()].argmax(1).cpu().numpy()
            target = target.cpu().numpy()
            target[target == native['ignore_label']] = -1
            valid = (target >= 0) & (target < len(classes))
            cm += np.bincount(len(classes) * target[valid] + pred[valid], minlength=len(classes)**2).reshape(cm.shape)
            if writer:
                writer.add(scene, pred, target)
            print('Evaluate {}/{} {}'.format(index + 1, len(scenes), scene['id']), flush=True)
    if writer:
        return writer.finish()
    union = cm.sum(0) + cm.sum(1) - cm.diagonal()
    if not (union > 0).any():
        raise ValueError('Validation split has no annotated points')
    return 100.0 * float((cm.diagonal()[union > 0] / union[union > 0]).mean())


def _train(model, cfg, native, classes, scenes):
    import torch
    selection = checkpoint_selection(cfg)
    validation = select_scenes(cfg, 'validation')[1] if selection == 'best' else None

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(scenes)

        def __getitem__(self, index):
            return _prepare(scenes[index], native, True)

    loader = torch.utils.data.DataLoader(Dataset(), batch_size=int(native['batch_size']),
        shuffle=True, num_workers=int(cfg.get('num_worker', native['workers'])),
        pin_memory=True, collate_fn=_batch, drop_last=False,
        generator=torch.Generator().manual_seed(int(cfg.get('seed', 0))))
    rate = float(native['base_lr'])
    groups = [dict(params=[p for n, p in model.named_parameters() if 'transformer_block' not in n], lr=rate),
              dict(params=[p for n, p in model.named_parameters() if 'transformer_block' in n], lr=rate * native['transformer_lr_scale'])]
    optimizer = torch.optim.AdamW(groups, lr=rate, weight_decay=float(native['weight_decay']))
    steps = int(native['epochs']) * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1.0 - step / max(steps, 1)) ** native['power'])
    weights = torch.tensor(native['class_weight'], dtype=torch.float32, device='cuda')
    criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=int(native['ignore_label']))
    amp = bool(native['use_amp'])
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
    history, output = [], Path(cfg['output'])
    for epoch in range(start, int(native['epochs']) + 1):
        model.train()
        total, count = 0.0, 0
        for packed in loader:
            # Upstream adds a random integer offset of 0 or 1 voxel per axis.
            packed[0][:, 1:] += (torch.rand(3) * 2).to(packed[0].dtype)
            optimizer.zero_grad(set_to_none=True)
            if not (packed[3] != native['ignore_label']).any():
                continue
            with torch.cuda.amp.autocast(enabled=amp):
                logits, target = _forward(model, packed)
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss at epoch {}'.format(epoch))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if native.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), native['max_grad_norm'])
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler lowers its scale when overflow skips the optimizer step.
            if scaler.get_scale() < scale:
                continue
            scheduler.step()
            total += float(loss.detach())
            count += 1
        if not count:
            raise RuntimeError('No optimizer updates completed at epoch {}'.format(epoch))
        score = _evaluate(model, validation, classes, native, cfg) if validation else None
        improved = score is not None and score > best
        if score is not None:
            best = max(best, score)
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'epoch': epoch, 'best_mIoU': best if validation else None, 'config': cfg,
            'manifest_sha256': file_hash(cfg['manifest'])}
        torch.save(checkpoint, str(output / 'last.pt'))
        if improved or (selection == 'last' and epoch == int(native['epochs'])):
            torch.save(checkpoint, str(output / 'model.pt'))
        history.append({'epoch': epoch, 'loss': total / count, 'validation_mIoU': score})
        write_json(output / 'history.json', history)
        print('Epoch {} loss {:.5f} validation mIoU {}'.format(epoch, total / count, score), flush=True)
    if not (output / 'model.pt').is_file():
        raise RuntimeError('Training did not produce the selected checkpoint')
    return {'checkpoint': str(output / 'model.pt'), 'best_validation_mIoU': best if validation else None, 'checkpoint_selection': selection}


def execute(action, cfg, root):
    if action == 'features':
        raise ValueError('Feature extraction requires PointNet++ trained on S3DIS')
    if int(cfg.get('num_gpus', 1)) != 1:
        raise ValueError('The SphereFormer adapter requires num_gpus=1')
    native = _configuration(cfg, root)
    classes, scenes = select_scenes(cfg, 'train' if action == 'train' else 'evaluate')
    if len(classes) != 19:
        raise ValueError('SphereFormer SemanticKITTI checkpoint expects 19 train classes')
    write_json(Path(cfg['output']) / 'sphereformer_config.json', native)
    model = _model(native)
    if action == 'train':
        return _train(model, cfg, native, classes, scenes)
    load_checkpoint(model, checkpoint_path(cfg))
    return _evaluate(model, scenes, classes, native, cfg, save=True)
