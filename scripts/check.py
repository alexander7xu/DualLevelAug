#!/usr/bin/env python3
import ast
from collections import Counter
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src'
PACKAGE = SOURCE / 'duallevelaug'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE))


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    ensure((PACKAGE / '__init__.py').is_file(),
           'Expected the public package at src/duallevelaug/')
    files = sorted(path for folder in (SOURCE, ROOT / 'scripts')
                   for path in folder.rglob('*.py'))
    for path in files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    legacy_files = sorted((PACKAGE / 'backends').glob('*.py'))
    for path in legacy_files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path),
                  feature_version=(3, 7))

    configs = sorted((ROOT / 'configs').glob('*.json'))
    ensure(configs, 'No example configurations found')
    parsed_configs = {
        path.name: json.loads(path.read_text(encoding='utf-8'))
        for path in configs
    }
    from duallevelaug.config import load_config
    from duallevelaug import experiments
    from duallevelaug.backends import build_command
    from duallevelaug.cli import parser
    from duallevelaug.data import map_label, split_full_scenes, validate_manifest
    from duallevelaug.reporting import computed_metrics
    from duallevelaug.resources import read_resource

    for path in configs:
        if path.name not in ('experiments.json', 'local.example.json'):
            load_config(path)

    catalog = experiments.catalog()
    catalog_ids = [entry['id'] for entry in catalog]
    ensure(len(catalog_ids) == len(set(catalog_ids)), 'Duplicate experiment IDs')
    ensure(parsed_configs['experiments.json']['experiments'] == catalog,
           'configs/experiments.json is inconsistent with the implemented catalog')
    coverage = experiments.coverage()
    ensure(coverage['table2']['cells'] == 24, 'Table 2 must cover 24 cells')
    ensure(sum(row['cells'] for key, row in coverage.items()
               if key.startswith('table')) == 223,
           'Tables must cover 223 result cells')
    ensure(all('figure%d' % index in coverage for index in range(1, 12)),
           'Figures 1 through 11 must have entries')

    locks = read_resource('backends.json')
    ensure(set(locks['backends']) == {'pointcept', 'pointnext', 'sphereformer'},
           'Unexpected backend lock inventory')
    for name, spec in {**locks['backends'], **locks.get('optional_sources', {})}.items():
        ensure(re.fullmatch(r'[0-9a-f]{40}', spec['commit']) is not None,
               'Backend source must have a full pinned commit: ' + name)
    reference = read_resource('paper_values.json')
    cell_ids = {cell['id'] for entry in catalog for cell in entry['cells']}
    ensure(cell_ids == set(reference['cells']),
           'Published reference cells differ from the experiment catalog')

    # Planning must not launch subprocesses.
    with patch('subprocess.run', side_effect=AssertionError('Unexpected process launch')), \
            patch('subprocess.check_output', side_effect=AssertionError('Unexpected process launch')):
        tasks = experiments.plan('all', root=ROOT)
        seen = set()
        commands = 0
        for task in tasks:
            ensure(set(task['dependencies']).issubset(seen),
                   'Task dependencies are not topologically ordered: ' + task['id'])
            ensure(task['id'] not in seen, 'Duplicate task ID: ' + task['id'])
            seen.add(task['id'])
            ensure(task['argv'][1:3] == ['-m', 'duallevelaug'],
                   'Planned commands must use the public package name')
            if task['stage'] in ('train', 'evaluate', 'features'):
                argv = build_command(task['stage'], task['config_path'], task['config'])
                ensure(Path(argv[1]).is_file(), 'Backend worker script is missing')
                ensure(argv[-3:] == [task['stage'], '--config', str(Path(task['config_path']).resolve())],
                       'Backend command does not preserve the selected action/configuration')
                commands += 1

    ensure(parser().parse_args(['experiments', 'plan', 'table2']).operation == 'plan',
           'Experiment CLI parser contract changed')
    ensure(map_label('semantickitti', 252) == map_label('semantickitti', 10),
           'Moving/static SemanticKITTI car classes must share a training ID')
    ensure(map_label('semantickitti', 0) == -1, 'Unlabeled points must be ignored')
    split = split_full_scenes([str(i) for i in range(10)], seed=0)
    ensure(list(split.values()).count('train') == 8 and
           list(split.values()).count('test') == 2,
           'Full-scene in-domain protocol must use an 80/20 split')
    try:
        validate_manifest({'dataset': 's3dis', 'scenes': [
            {'id': 'tile_a', 'source_id': 'room', 'split': 'train'},
            {'id': 'tile_b', 'source_id': 'room', 'split': 'test'},
        ]})
    except ValueError:
        pass
    else:
        raise ValueError('Manifest validation accepted full-scene split leakage')
    metrics = computed_metrics({'confusion_matrix': [[2, 1], [0, 1]],
                                'class_names': ['a', 'b']})
    ensure(metrics['allacc'] == 75.0 and abs(metrics['miou'] - 175.0 / 3) < 1e-10,
           'Confusion-matrix metric computation changed')

    summary = {
        'syntax_files': len(files),
        'legacy_syntax_files': len(legacy_files),
        'json_configs': len(configs),
        'packaged_resources': ['backends.json', 'paper_values.json'],
        'experiment_conditions': len(catalog),
        'result_cells': len(cell_ids),
        'planned_tasks': dict(Counter(task['stage'] for task in tasks)),
        'backend_commands': commands,
        'protocol_checks': 'passed',
        'numerical_validation': 'not performed by this standard-library check',
        'success': True,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AssertionError, ImportError, KeyError, OSError, SyntaxError, ValueError) as exc:
        print('Source validation failed: %s' % exc, file=sys.stderr)
        raise SystemExit(1)
