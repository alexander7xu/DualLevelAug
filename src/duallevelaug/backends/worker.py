#!/usr/bin/env python
"""Run a backend without importing the Python 3.10 controller."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

from common import verify_backend, file_hash, write_json, seed_all, checkpoint_selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'evaluate', 'features'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    cfg['checkpoint_selection'] = checkpoint_selection(cfg)
    name, source, lock = verify_backend(cfg)
    try:
        import numpy
        import torch
    except ImportError as exc:
        raise RuntimeError('Backend environment lacks NumPy/PyTorch; use config.python for the documented CUDA environment') from exc
    if torch.__version__.split('+')[0] != lock['torch'].split('+')[0] or torch.version.cuda != lock['cuda']:
        raise RuntimeError('Torch/CUDA version mismatch: expected {}/{}, got {}/{}'.format(lock['torch'], lock['cuda'], torch.__version__, torch.version.cuda))
    if not torch.cuda.is_available():
        raise RuntimeError('Backend execution requires Linux and an NVIDIA CUDA device')
    output = Path(cfg['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config_snapshot.json', cfg)
    provenance = {'backend': name, 'backend_root': str(source), 'commit': lock['commit'], 'submodules': lock.get('submodules', {}),
                  'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__,
                  'cuda': torch.version.cuda, 'seed': int(cfg.get('seed', 0)), 'action': args.action,
                  'manifest_sha256': file_hash(cfg['manifest']),
                  'data_split_hash': json.loads(Path(cfg['manifest']).read_text()).get('split_hash'),
                  'generated_manifest_sha256': file_hash(cfg['generated_manifest']) if cfg.get('generated_manifest') else None}
    write_json(output / 'provenance.json', provenance)
    # Load local adapters before upstream modules can shadow them.
    module = __import__({'pointcept': 'pointcept_backend', 'pointnext': 'openpoints_backend',
                         'sphereformer': 'sphereformer_backend'}[name])
    sys.path.insert(0, str(source))
    seed_all(int(cfg.get('seed', 0)))
    result = module.execute(args.action, cfg, source)
    provenance['completed'] = True
    if (output / 'model.pt').is_file():
        provenance['checkpoint_sha256'] = file_hash(output / 'model.pt')
    provenance['pip_freeze'] = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], universal_newlines=True).splitlines()
    write_json(output / 'provenance.json', provenance)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        sys.stderr.write('DualLevelAug backend error: {}\n'.format(exc))
        sys.exit(2)
