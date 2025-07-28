from pathlib import Path
import os
import subprocess
import sys

MODELS = {
    'ptv3': 'pointcept', 'octformer': 'pointcept', 'ptv2': 'pointcept',
    'minkunet': 'pointcept', 'pointnext-xl': 'pointnext',
    'pointnet2': 'pointnext', 'sphereformer': 'sphereformer',
}
SUPPORTED = {
    'ptv3': ('scannet', 's3dis', 'semantickitti', 'stpls3d'),
    'octformer': ('scannet', 's3dis', 'semantickitti'),
    'ptv2': ('scannet', 's3dis', 'semantickitti'),
    'minkunet': ('scannet', 's3dis', 'semantickitti'),
    'pointnext-xl': ('scannet', 's3dis'),
    'pointnet2': ('s3dis',), 'sphereformer': ('semantickitti',),
}


def normalize_dataset(value):
    return str(value).lower().replace('_', '').replace('-', '')


def backend_name(cfg):
    model = cfg['model'].lower()
    if model not in MODELS:
        raise ValueError('Unsupported backbone: ' + model)
    if normalize_dataset(cfg['dataset']) not in SUPPORTED[model]:
        raise ValueError('Unsupported dataset/model: {}/{}'.format(cfg['dataset'], model))
    return MODELS[model]


def build_command(action, config_path, cfg):
    if action not in ('train', 'evaluate', 'features'):
        raise ValueError('Unsupported backend action: ' + action)
    backend = backend_name(cfg)
    if action == 'features' and cfg['model'] != 'pointnet2':
        raise ValueError('Paper feature descriptor requires model=pointnet2')
    settings = cfg.get('backends', {}).get(backend, {})
    executable = str(cfg.get('python', settings.get('python', sys.executable)))
    return [executable, str(Path(__file__).with_name('worker.py')), action,
            '--config', str(Path(config_path).resolve())]


def run(action, config_path, cfg):
    argv = build_command(action, config_path, cfg)
    env = dict(os.environ)
    env['PYTHONNOUSERSITE'] = '1'
    # Controller paths can load incompatible libraries in Python 3.7 workers.
    env.pop('PYTHONPATH', None)
    result = subprocess.run(argv, env=env, check=True)
    return {'command': argv, 'returncode': result.returncode,
            'output': str(Path(cfg['output']).resolve())}
