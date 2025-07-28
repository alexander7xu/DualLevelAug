#!/usr/bin/env python3
"""Run with Python 3.10+; --python selects the backend environment."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

if sys.version_info < (3, 10):
    raise SystemExit('Run setup_backends.py with Python 3.10+; use --python to select a legacy backend interpreter.')

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Python 3.10 requires tomli to read pyproject.toml. Install the controller package '
            'or run: python -m pip install tomli==2.2.1'
        ) from exc

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / 'src/duallevelaug/resources/backends.json').read_text())
with (ROOT / 'pyproject.toml').open('rb') as handle:
    PROFILES = tomllib.load(handle)['tool']['duallevelaug']['backends']


def profile_for(name):
    spec, profile = LOCK['backends'][name], PROFILES[name]
    if spec['dependency_profile'] != 'tool.duallevelaug.backends.' + name:
        raise ValueError('Source lock names an unexpected dependency profile: ' + name)
    for field in ('python', 'cuda'):
        if profile[field] != spec[field]:
            raise ValueError('Backend profile and source lock disagree on {}: {}'.format(field, name))
    requirements = profile['requirements']
    pins = {}
    for requirement in requirements:
        package, separator, version = requirement.partition('==')
        if not separator or not package or not version:
            raise ValueError('Backend dependencies must have exact version pins: ' + requirement)
        if package in pins:
            raise ValueError('Duplicate backend dependency: ' + package)
        pins[package] = requirement
    if pins.get('torch') != 'torch==' + spec['torch']:
        raise ValueError('Backend profile and runtime lock disagree on Torch: ' + name)
    bootstrap = [pins[package] for package in profile['bootstrap']]
    return profile, bootstrap


def commands(name, external, python):
    spec = LOCK['backends'][name]
    profile, bootstrap = profile_for(name)
    location = external / name
    fetch = [(['git', 'clone', '--no-checkout', spec['url'], str(location)], ROOT),
             (['git', '-C', str(location), 'checkout', '--detach', spec['commit']], ROOT)]
    if spec.get('submodules'):
        fetch.append((['git', '-C', str(location), '-c', 'url.https://github.com/.insteadOf=git@github.com:',
                       'submodule', 'update', '--init', '--recursive'], ROOT))
    for extra in spec.get('optional_sources', []):
        item = LOCK['optional_sources'][extra]
        fetch.extend([(['git', 'clone', '--no-checkout', item['url'], str(external / extra)], ROOT),
                      (['git', '-C', str(external / extra), 'checkout', '--detach', item['commit']], ROOT)])
    # Build-time imports (notably flash-attn) need Torch installed before the full profile.
    install = [([python, '-m', 'pip', 'install', *profile['bootstrap_pip_options'], *bootstrap], ROOT),
               ([python, '-m', 'pip', 'install', '--no-build-isolation',
                 *profile['pip_options'], *profile['requirements']], ROOT)]
    for extra in spec.get('optional_sources', []):
        install.append(([python, '-m', 'pip', 'install', '--no-build-isolation',
                         str(external / extra)], external / extra))
    for extension in spec.get('extensions', []):
        install.append(([python, '-m', 'pip', 'install', '--no-build-isolation',
                         str(location / extension)], location / extension))
    return fetch, install


def verify(name, external):
    spec = LOCK['backends'][name]
    checks = [(external / name, spec['commit'])]
    checks += [(external / name / child, commit) for child, commit in spec.get('submodules', {}).items()]
    checks += [(external / extra, LOCK['optional_sources'][extra]['commit']) for extra in spec.get('optional_sources', [])]
    for path, expected in checks:
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise RuntimeError('Wrong commit in {}: {} != {}'.format(path, actual, expected))
    for tree, expected in spec.get('vendored_trees', {}).items():
        actual = subprocess.check_output(['git', '-C', str(external / name), 'rev-parse', 'HEAD:' + tree], text=True).strip()
        if actual != expected:
            raise RuntimeError('Wrong vendored tree: ' + tree)
    return checks


def verify_interpreter(name, python):
    expected = PROFILES[name]['python'].split('.')
    actual = subprocess.check_output(
        [python, '-c', 'import platform; print(platform.python_version())'], text=True
    ).strip()
    if actual.split('.')[:len(expected)] != expected:
        raise RuntimeError('Backend {} requires Python {}, but {} reports {}'.format(
            name, PROFILES[name]['python'], python, actual))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=['all'] + list(LOCK['backends']), nargs='?', default='all')
    parser.add_argument('--external', default=str(ROOT / 'external'))
    parser.add_argument('--python', default=sys.executable, help='Target backend interpreter; setup itself runs with controller Python 3.10+')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fetch', action='store_true', help='Clone pinned source repositories')
    mode.add_argument('--install', action='store_true', help='Install the dependency profile and CUDA kernels into --python')
    mode.add_argument('--verify', action='store_true', help='Verify checkout and submodule revisions')
    args = parser.parse_args(argv)
    names = list(LOCK['backends']) if args.backend == 'all' else [args.backend]
    external = Path(args.external).resolve()
    if args.install and len(names) != 1:
        parser.error('Install one backend at a time in its isolated environment')
    for name in names:
        fetch, install = commands(name, external, args.python)
        if args.verify:
            verify(name, external)
            print(name + ': all source revisions verified')
            continue
        if args.install:
            verify_interpreter(name, args.python)
            verify(name, external)
        chosen = fetch if args.fetch else install if args.install else fetch + install
        for command, directory in chosen:
            print('[{}] {}  (cwd={})'.format(name, shlex.join(command), directory), flush=True)
            if args.fetch or args.install:
                external.mkdir(parents=True, exist_ok=True)
                if command[:2] == ['git', 'clone'] and Path(command[-1]).exists():
                    raise FileExistsError('Destination already exists; verify it instead: ' + command[-1])
                subprocess.run(command, cwd=str(directory), check=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyError, ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.stderr.write('Backend setup error: {}\n'.format(exc))
        sys.exit(2)
