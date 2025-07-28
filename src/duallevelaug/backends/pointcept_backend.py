import copy
import importlib
from pathlib import Path
import shutil

try:
    from .common import load_scene, safe_id, select_scenes, checkpoint_path, PredictionWriter, checkpoint_selection
except ImportError:  # Direct execution by worker.py.
    from common import load_scene, safe_id, select_scenes, checkpoint_path, PredictionWriter, checkpoint_selection


_REGISTERED = False

CONFIGS = {
    'ptv3': 'semseg-pt-v3m1-0-base.py',
    'ptv2': 'semseg-pt-v2m2-0-base.py',
    'minkunet': 'semseg-minkunet34c-0-base.py',
    'octformer': 'semseg-octformer-v1m1-0-base.py',
}


def config_recipe(model, dataset):
    dataset = dataset.lower().replace('_', '').replace('-', '')
    canonical = 'semantic_kitti' if dataset == 'semantickitti' else dataset
    if model == 'octformer':
        source = 'scannet'
    elif canonical == 'semantic_kitti' and model == 'ptv3':
        source = 'scannet'
    elif canonical == 'stpls3d':
        source = 's3dis'
    else:
        source = canonical
    recipe = {'source': 'configs/{}/{}'.format(source, CONFIGS[model]), 'dataset': canonical,
              'adapted': source != canonical}
    if canonical == 'semantic_kitti' and model in ('ptv3', 'octformer'):
        recipe['data_source'] = 'configs/semantic_kitti/semseg-pt-v2m2-0-base.py'
    return recipe


def _walk_transforms(value, fn):
    if isinstance(value, dict):
        if 'type' in value:
            fn(value)
        for child in value.values():
            _walk_transforms(child, fn)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _walk_transforms(child, fn)


def configure(cfg, root, action):
    from pointcept.utils.config import Config
    model, dataset = cfg['model'], cfg['dataset']
    recipe = config_recipe(model, dataset)
    native = Config.fromfile(str(root / recipe['source']))
    if 'data_source' in recipe:
        native.data = copy.deepcopy(Config.fromfile(str(root / recipe['data_source'])).data)
    classes, _ = select_scenes(cfg, 'train' if action == 'train' else 'evaluate')
    count = len(classes)
    native.data.num_classes, native.data.names = count, classes
    if 'num_classes' in native.model:
        native.model.num_classes = count
    if 'num_classes' in native.model.backbone:
        native.model.backbone.num_classes = count
    canonical = recipe['dataset']
    if recipe['adapted'] and canonical == 'semantic_kitti':
        native.model.backbone.in_channels = 4
        native.epoch = native.eval_epoch = 50
        if model == 'ptv3':
            native.optimizer.lr = 0.002
            native.scheduler.max_lr = [0.002, 0.0002]
            native.param_dicts = [dict(keyword='block', lr=0.0002)]
        else:
            native.model.backbone.octree_scale_factor = 64.0
            native.model.backbone.octree_depth = 12
    if canonical == 'stpls3d':
        native.epoch = native.eval_epoch = 100
        _walk_transforms(native.data, lambda t: t.update(grid_size=0.1) if t.get('type') == 'GridSample' else None)
    if model == 'octformer':
        # OctFormer requests normals separately even when they are not features.
        def normal_keys(transform):
            if transform.get('type') == 'Collect':
                transform['keys'] = tuple(dict.fromkeys(tuple(transform['keys']) + ('normal',)))
        _walk_transforms(native.data, normal_keys)
        native.mix_prob = 0.8 if cfg.get('variant') == 'mix3d' else 0.0
    elif cfg.get('variant') == 'mix3d':
        native.mix_prob = 0.8
    elif 'mix3d' in cfg:
        native.mix_prob = 0.8 if cfg['mix3d'] else 0.0
    # With no explicit override PTv3 keeps all upstream baseline augmentations.
    native.seed = int(cfg.get('seed', 0))
    native.save_path = str(Path(cfg['output']).resolve())
    native.enable_wandb = False
    selection = checkpoint_selection(cfg)
    if action == 'train' and selection == 'last':
        native.evaluate = False
        native.hooks = [h for h in native.hooks if h['type'] not in ('SemSegEvaluator', 'PreciseEvaluator')]
    recipe['checkpoint_selection'] = selection
    native.weight = checkpoint_path(cfg) if action != 'train' else cfg.get('checkpoint')
    native.resume = bool(cfg.get('resume', False))
    overrides = dict(cfg.get('training', {}))
    for key in ('epoch', 'eval_epoch', 'batch_size', 'num_worker', 'enable_amp', 'clip_grad'):
        if key in cfg:
            overrides[key] = cfg[key]
        if key in overrides:
            native[key] = overrides[key]
    if 'epochs' in overrides:
        native.epoch = native.eval_epoch = int(overrides['epochs'])
    if native.epoch % native.eval_epoch:
        raise ValueError('Pointcept epoch must be divisible by eval_epoch; set both for short validation runs')
    if cfg.get('upstream_overrides'):
        native.merge_from_dict(cfg['upstream_overrides'])
    for purpose, selector in [('train', 'train'), ('val', 'validation'), ('test', 'evaluate')]:
        section = native.data[purpose]
        section['type'] = 'DualLevelManifestDataset'
        section['manifest_config'] = cfg
        section['purpose'] = selector
        section['require_normal'] = model == 'octformer' or canonical == 'scannet'
        section['split'] = 'train' if purpose == 'train' else 'val'
        section['data_root'] = str(Path(cfg['manifest']).resolve().parent)
        # Dataset-specific kwargs are not accepted by the common DefaultDataset.
        for key in ('la_file', 'lr_file', 'class2id', 'test_area'):
            section.pop(key, None)
    native.data.train.loop = native.epoch // native.eval_epoch
    return native, recipe


def register_dataset():
    global _REGISTERED
    if _REGISTERED:
        return
    from pointcept.datasets.defaults import DefaultDataset
    from pointcept.datasets.builder import DATASETS

    @DATASETS.register_module()
    class DualLevelManifestDataset(DefaultDataset):
        def __init__(self, manifest_config, purpose, require_normal=False, **kwargs):
            self.manifest_config = manifest_config
            self.purpose = purpose
            self.require_normal = require_normal
            super().__init__(**kwargs)

        def get_data_list(self):
            _, scenes = select_scenes(self.manifest_config, self.purpose)
            return scenes

        def get_data_name(self, idx):
            return safe_id(self.data_list[idx % len(self.data_list)]['id'])

        def get_split_name(self, idx):
            return self.data_list[idx % len(self.data_list)]['split']

        def get_data(self, idx):
            scene = self.data_list[idx % len(self.data_list)]
            data = load_scene(scene, self.require_normal)
            # Upstream transforms accept only these point-array keys.
            result = {key: data[key] for key in ('coord', 'color', 'normal', 'strength', 'segment', 'instance') if key in data}
            result.update(name=self.get_data_name(idx), split=self.get_split_name(idx))
            return result
    _REGISTERED = True
    return DualLevelManifestDataset


def _run_native(native, action):
    register_dataset()
    if native.model.backbone.type == 'OctFormer-v1m1':
        importlib.import_module('pointcept.models.octformer')
    if native.model.backbone.type.startswith('Mink'):
        importlib.import_module('pointcept.models.sparse_unet.mink_unet')
    from pointcept.engines.defaults import default_setup
    native = default_setup(native)
    if action == 'train':
        from pointcept.engines.train import TRAINERS
        trainer = TRAINERS.build(dict(type=native.train.type, cfg=native))
        trainer.train()
    else:
        from pointcept.engines.test import TESTERS
        tester = TESTERS.build(dict(type=native.test.type, cfg=native))
        tester.test()


def execute(action, cfg, root):
    import numpy as np
    try:
        from .common import write_json
    except ImportError:
        from common import write_json
    register_dataset()
    # Upstream __init__ does not import these optional architectures.
    if cfg['model'] == 'octformer':
        importlib.import_module('pointcept.models.octformer')
    if cfg['model'] == 'minkunet':
        importlib.import_module('pointcept.models.sparse_unet.mink_unet')
    native, recipe = configure(cfg, root, action)
    output = Path(cfg['output']).resolve()
    (output / 'model').mkdir(parents=True, exist_ok=True)
    native.dump(str(output / 'pointcept_config.py'))
    write_json(output / 'backend_recipe.json', recipe)
    if action != 'train':
        # SemSegTester cache keys omit the checkpoint, so cached predictions may be stale.
        _, scenes = select_scenes(cfg, 'evaluate')
        for scene in scenes:
            cache = output / 'result' / (safe_id(scene['id']) + '_pred.npy')
            if cache.is_file():
                cache.unlink()
    num_gpus = int(cfg.get('num_gpus', 1))
    if num_gpus > 1:
        from pointcept.engines.launch import launch
        # Each spawned process needs its own dataset registration.
        launch(_run_native, num_gpus_per_machine=num_gpus, num_machines=1,
               machine_rank=0, dist_url='auto', cfg=(native, action))
    else:
        _run_native(native, action)
    if action == 'train':
        filename = 'model_last.pth' if checkpoint_selection(cfg) == 'last' else 'model_best.pth'
        best = output / 'model' / filename
        if not best.is_file():
            raise RuntimeError('Pointcept did not produce requested checkpoint: ' + filename)
        shutil.copyfile(str(best), str(output / 'model.pt'))
        return {'checkpoint': str(output / 'model.pt')}
    classes, scenes = select_scenes(cfg, 'evaluate')
    writer = PredictionWriter(output, classes, cfg)
    for scene in scenes:
        prediction = np.load(str(output / 'result' / (safe_id(scene['id']) + '_pred.npy')))
        writer.add(scene, prediction, load_scene(scene)['segment'])
    return writer.finish()
