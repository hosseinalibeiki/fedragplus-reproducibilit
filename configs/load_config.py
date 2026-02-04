"""
Config loader with YAML includes + deep-merge.

Usage:
  python -m configs.load_config --config configs/fedragplus.yaml --out merged.yaml
  python -m configs.load_config --config configs/fedragplus.yaml --print
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries.

    - dict values are merged
    - lists are replaced (override wins)
    - scalars are replaced (override wins)
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping (dict). Got: {type(data)} in {path}")
    return data


def _resolve_includes(cfg: Dict[str, Any], cfg_path: Path, seen: List[Path]) -> Dict[str, Any]:
    """Resolve 'include' key (list of YAML files) with deep-merge semantics.

    Included files are merged first (in listed order), then the current file overrides.
    """
    includes = cfg.get("include", []) or []
    if isinstance(includes, (str, Path)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"'include' must be a list of paths in {cfg_path}")

    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = (cfg_path.parent / str(inc)).resolve()
        if inc_path in seen:
            chain = " -> ".join([str(p) for p in seen + [inc_path]])
            raise ValueError(f"Detected circular include: {chain}")
        seen_next = seen + [inc_path]
        inc_cfg = _read_yaml(inc_path)
        inc_cfg = _resolve_includes(inc_cfg, inc_path, seen_next)
        merged = _deep_merge(merged, inc_cfg)

    # Remove include key from current config before merge
    cfg_no_inc = dict(cfg)
    cfg_no_inc.pop("include", None)
    merged = _deep_merge(merged, cfg_no_inc)
    return merged


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = _read_yaml(config_path)
    return _resolve_includes(cfg, config_path, seen=[config_path])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a YAML config file (may contain include: [...])")
    ap.add_argument("--out", default=None, help="Optional path to write merged YAML")
    ap.add_argument("--print", action="store_true", help="Print merged YAML to stdout")
    args = ap.parse_args()

    merged = load_config(args.config)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, sort_keys=False)

    if args.print or not args.out:
        print(yaml.safe_dump(merged, sort_keys=False))


if __name__ == "__main__":
    main()
