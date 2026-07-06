"""Hyperparameter search spaces for weak-supervision sweeps (train-from-cache).

Sweeps ILR + iterated-model knobs. Scoring uses each run's ``final_assignments.csv``
hit rate (see ``assignment_score.score_final_assignments_hit_rate``).

Blur / Laplacian ``threshold`` is fixed at 50 for eval alignment (not swept here).
"""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

from ..core.config import DEFAULT_CONFIG

# Fixed preprocessing / eval alignment (not varied by this sweep module).
SWEEP_FIXED_THRESHOLD = 50.0

# Keys swept by ``generate_configs`` (order stable for CSV columns).
# Defaults are reasonable ranges; override per-axis with CLI ``--sweep-*`` comma lists.
PARAM_GRID: Dict[str, List[Any]] = {
    "ilr_epochs": [1000],
    "n_components": [17],
    "triplet_margin": [0.1],
    "refinement_loops": [5],
    "adapter_epochs": [5],
    "adapter_lr": [1e-4],
}

SWEEP_PARAM_KEYS: Tuple[str, ...] = tuple(PARAM_GRID.keys())


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(float(s)))
    if not out:
        raise ValueError(f"No integers parsed from: {raw!r}")
    return out


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError(f"No floats parsed from: {raw!r}")
    return out


def build_param_grid_from_cli(overrides: Dict[str, Optional[str]]) -> Dict[str, List[Any]]:
    """
    Start from :data:`PARAM_GRID` defaults; replace any axis whose CLI value is a
    non-empty string (comma-separated list).

    *overrides* maps logical key -> raw string or None.
    ``triplet_margin`` and ``adapter_lr`` use float parsing; others use int.
    """
    grid: Dict[str, List[Any]] = {k: list(v) for k, v in PARAM_GRID.items()}
    float_keys = {"triplet_margin", "adapter_lr"}
    for key, raw in overrides.items():
        if key not in grid:
            continue
        if raw is None or not str(raw).strip():
            continue
        if key in float_keys:
            grid[key] = _parse_float_list(str(raw))
        else:
            grid[key] = _parse_int_list(str(raw))
    return grid


def _merge_into_base(overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def generate_configs(
    search_type: Literal["grid", "random"] = "random",
    num_samples: int = 50,
    *,
    random_state: Optional[int] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build full trainer config dicts (merged with DEFAULT_CONFIG).

    For ``random``, each sample draws one value per swept key independently.
    """
    grid = param_grid if param_grid is not None else PARAM_GRID
    keys = list(grid.keys())
    rng = random.Random(random_state)

    if search_type == "grid":
        value_lists = [grid[k] for k in keys]
        combos: List[Tuple[Any, ...]] = list(itertools.product(*value_lists))
        out: List[Dict[str, Any]] = []
        for tup in combos:
            overrides = dict(zip(keys, tup))
            out.append(_merge_into_base(overrides))
        return out

    if search_type != "random":
        raise ValueError(f"search_type must be 'grid' or 'random', got {search_type!r}")

    out_rand: List[Dict[str, Any]] = []
    for _ in range(max(1, num_samples)):
        overrides = {k: rng.choice(grid[k]) for k in keys}
        out_rand.append(_merge_into_base(overrides))
    return out_rand


def grid_size(param_grid: Optional[Dict[str, List[Any]]] = None) -> int:
    """Product of list lengths for a full grid search."""
    grid = param_grid if param_grid is not None else PARAM_GRID
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def describe_param_grid(param_grid: Optional[Dict[str, List[Any]]] = None) -> str:
    grid = param_grid if param_grid is not None else PARAM_GRID
    lines = [
        f"Fixed eval blur threshold (not swept): {SWEEP_FIXED_THRESHOLD}",
        "Swept keys (ILR + iterated adapter):",
    ]
    for k, vals in grid.items():
        lines.append(f"  {k}: {vals}")
    lines.append(f"Full grid size: {grid_size(grid)}")
    return "\n".join(lines)


__all__ = [
    "PARAM_GRID",
    "SWEEP_PARAM_KEYS",
    "SWEEP_FIXED_THRESHOLD",
    "build_param_grid_from_cli",
    "describe_param_grid",
    "generate_configs",
    "grid_size",
]
