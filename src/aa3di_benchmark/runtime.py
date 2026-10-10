# SPDX-License-Identifier: Apache-2.0
"""Configuration paths, immutable run identity and task provenance."""
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path
import yaml


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


class Run:
    def __init__(self, config, output):
        self.config_path = Path(config).resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        if not isinstance(self.config, dict) or not self.config.get('protocol'):
            raise ValueError('Configuration must declare a protocol')
        self.output = Path(output).resolve()

    def path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError('Required path is not configured')
        value = os.path.expandvars(os.path.expanduser(value))
        if re.search(r'\$[A-Za-z_{]', value):
            raise ValueError(f'Unset environment variable in path: {value}')
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.config_path.parent / path).resolve()

    def record(self, task, status, inputs=(), details=None, bindings=None):
        source = Path(__file__).parent
        frozen = {'config_sha256': identity(self.config_path)['sha256'],
                  'config': self.config,
                  'code': {str(p.relative_to(source)): identity(p)['sha256'] for p in sorted(source.rglob('*.py'))}}
        path = self.output / 'manifest.json'
        manifest = json.loads(path.read_text()) if path.exists() else {'frozen': frozen, 'tasks': {}, 'status': 'partial'}
        if manifest['frozen'] != frozen:
            raise ValueError('Config or code changed: choose a new output directory')
        old = manifest['tasks'].get(task)
        fingerprints = [identity(p) for p in inputs]
        registry = manifest.setdefault('bindings', {})
        by_path = {f['path']: f for f in fingerprints}
        for name, paths in (bindings or {}).items():
            values = [by_path.get(str(Path(p).resolve())) or identity(p) for p in paths]
            if name in registry and registry[name] != values:
                raise ValueError(f'Model identity changed across tasks: {name}; choose a new output directory')
            registry[name] = values
        artifacts = manifest.setdefault('artifacts', {})
        for fingerprint in fingerprints:
            key = fingerprint['path']
            if key in artifacts and artifacts[key] != fingerprint:
                raise ValueError(f'An input changed across tasks: {key}; choose a new output directory')
            artifacts[key] = fingerprint
        if old and old.get('inputs') != fingerprints:
            raise ValueError(f'Inputs changed for {task}: choose a new output directory')
        packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                    if dist.metadata.get('Name')}
        manifest['tasks'][task] = {'status': status, 'inputs': fingerprints, 'details': details,
                                  'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': packages}}
        write_json(path, manifest)
