# -*- coding: utf-8 -*-
"""Portable path resolution for RFGP configs.

Relative paths resolve against ``path.root`` (default: package directory).

Environment overrides:

- ``LOC_DATA_ROOT``: prefix for data root directories
- ``LOC_OUT_ROOT``: prefix for ``logdir``, fig paths, and ``test_pos_path``
"""
import os
from copy import deepcopy
from typing import Any, Dict, Optional

_OUT_KEYS = (
    'logdir',
    'train_fig_path',
    'test_fig_path',
    'test_pos_path',
    'pre_ckpts',
)

_PKG_ROOT = os.path.dirname(os.path.abspath(__file__))


def _abspath(path: str, base: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def _resolve_data_map(data_map: Optional[Dict[str, Any]], base: str, data_root: Optional[str]) -> Dict[str, str]:
    if not data_map:
        return {}
    resolved = {}
    for modality, value in data_map.items():
        if value in (None, ''):
            continue
        if data_root and not os.path.isabs(value):
            resolved[str(modality)] = _abspath(value, data_root)
        else:
            resolved[str(modality)] = _abspath(value, base)
    return resolved


def resolve_paths(
    path_cfg: Dict[str, Any],
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a copy of ``path_cfg`` with absolute, env-overridable paths."""
    cfg = deepcopy(path_cfg)

    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
    else:
        config_dir = _PKG_ROOT

    root = cfg.get('root') or _PKG_ROOT
    if not os.path.isabs(str(root)):
        root = _abspath(root, config_dir)
    else:
        root = os.path.normpath(root)

    data_env = os.environ.get('LOC_DATA_ROOT')
    out_env = os.environ.get('LOC_OUT_ROOT')

    cfg['train_data_roots'] = _resolve_data_map(
        cfg.get('train_data_roots'), root, data_env)
    cfg['test_data_roots'] = _resolve_data_map(
        cfg.get('test_data_roots'), root, data_env)

    # Backward-compatible single-dir keys (optional)
    for key in ('ble_datadir', 'rfid_datadir', 'wifi_datadir', 'iiot_datadir'):
        if key in cfg and cfg[key] not in (None, ''):
            value = cfg[key]
            cfg[key] = (
                _abspath(value, data_env)
                if data_env and not os.path.isabs(value)
                else _abspath(value, root)
            )

    for key in _OUT_KEYS:
        if key not in cfg or cfg[key] in (None, ''):
            continue
        value = cfg[key]
        if out_env and not os.path.isabs(value):
            cfg[key] = _abspath(value, out_env)
        else:
            cfg[key] = _abspath(value, root)

    cfg['root'] = root
    return cfg


def default_config_path() -> str:
    """Return the default ``configs/pre.yaml`` path next to this package."""
    return os.path.join(_PKG_ROOT, 'configs', 'pre.yaml')


def merge_renderer_cfg(renderer_kwargs: Dict[str, Any], scene_id: int) -> Dict[str, Any]:
    """Build per-scene renderer kwargs from defaults + optional overrides.

    Expected YAML shape::

        renderer:
          mode: spectrum
          scale_worldsize: 1
          defaults:
            near: 0
            far: 15
            n_samples: 16
          overrides:
            11: {far: 25}
            14: {far: 5}
    """
    cfg = {
        'scale_worldsize': renderer_kwargs.get('scale_worldsize', 1),
        'mode': renderer_kwargs.get('mode', 'spectrum'),
        'near': 0,
        'far': 15,
        'n_samples': 16,
    }
    defaults = renderer_kwargs.get('defaults') or {}
    cfg.update(defaults)

    overrides = renderer_kwargs.get('overrides') or {}
    scene_override = (
        overrides.get(scene_id)
        or overrides.get(str(scene_id))
    )
    if scene_override:
        cfg.update(scene_override)
    return cfg
