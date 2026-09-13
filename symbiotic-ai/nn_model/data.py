"""Data loaders for features, boundary labels, and object-count ground truth."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .labels_io import read_label_file, write_label_file
from .states import IDX_TO_STATE, STATE_TO_IDX, STATES

FeatureSource = Literal["processed_3d", "csv_outputs"]

REPO_ROOT = Path(__file__).resolve().parents[1]
HMM_TESTING = REPO_ROOT / "hmm-testing"
DEFAULT_FEATURES_3D = HMM_TESTING / "processed_features_3d"
DEFAULT_CSV_OUTPUTS = REPO_ROOT / "objectdetector" / "csv_outputs"
DEFAULT_LABELS = HMM_TESTING / "csv_labels"
DEFAULT_GROUND_TRUTH = REPO_ROOT / "ground_truth.csv"
DEFAULT_PICKLIST_JSONS = HMM_TESTING / "picklist_jsons"
DEFAULT_PICKLIST_LABELS = HMM_TESTING / "picklist_labels"
DEFAULT_VIDEO_DIR = HMM_TESTING / "picklist_videos"

FEATURE_NAMES_3D = ("p_hold", "direction", "net_aruco")
FEATURE_NAMES_CSV = ("p_hold", "direction", "net_aruco", "pick_count", "place_count")

# One item = one pass through this cycle.
CYCLE_TOKENS: Tuple[str, ...] = ("m", "a", "e", "i")

NUM_STATES = len(STATES)
BLANK_INDEX = NUM_STATES
NUM_CLASSES = NUM_STATES + 1
STATE_OFFSET = 0  # ctc_class_index = state_index + STATE_OFFSET (blank-last)


@dataclass
class PicklistSample:
    picklist_id: str
    features: np.ndarray  # (T, D)
    labels: Optional[np.ndarray]  # (T,) dense per-frame state indices, or None
    targets: Optional[np.ndarray]  # (U,) CTC target sequence, or None
    fps: float
    object_count: Optional[int]


def picklist_id_from_stem(stem: str) -> str:
    m = re.match(r"picklist_(\d+)$", stem, re.I)
    if not m:
        raise ValueError(f"Unexpected picklist stem: {stem!r}")
    return m.group(1)


def parse_boundary_labels(label_path: Path) -> Tuple[List[Tuple[int, str]], List[str]]:
    """Parse a boundary label file -> (boundaries, tokens).

    Delegates to labels_io.read_label_file, which handles both the quoted-tab
    csv_labels format and the frame_index,code format bulk_predict writes.
    """
    return read_label_file(label_path)


def _token_to_state_name(token: str) -> Optional[str]:
    from .states import TOKEN_TO_STATE

    if token in TOKEN_TO_STATE and token != "sil":
        return TOKEN_TO_STATE[token]
    return None


def tokens_to_target_sequence(
    tokens: Sequence[str],
    drop_trailing_cycle_start: bool = True,
) -> np.ndarray:
    """Token list -> (U,) CTC target sequence of state indices.

    Drops "sil" and unknown tokens: silence is the blank's job under CTC and
    must not appear in the target sequence.

    Annotators consistently close a picklist with a trailing "m", which makes
    the sequence n+1 cycle-starts for n items rather than a clean n x maei.
    With drop_trailing_cycle_start (the default) that final "m" is cropped, so
    training targets match the n x maei structure the JSON item counts imply.
    Adjacent duplicates are otherwise preserved: two runs of the same state
    separated by a dropped sil are genuinely two emissions.
    """
    kept_tokens: List[str] = []
    kept_idx: List[int] = []
    for tok in tokens:
        name = _token_to_state_name(tok)
        if name is None:
            continue
        kept_tokens.append(tok)
        kept_idx.append(STATE_TO_IDX[name])

    if (
        drop_trailing_cycle_start
        and len(kept_idx) > 1
        and kept_tokens[-1] == CYCLE_TOKENS[0]
    ):
        kept_idx.pop()
        kept_tokens.pop()

    return np.asarray(kept_idx, dtype=np.int64)


def count_picklist_items(json_path: Path) -> int:
    """Total items in a picklist_NNN.json, summed across its sublists."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entries = data.get("picklists")
    if entries is None:
        raise ValueError(f"{json_path}: no 'picklists' key")
    total = 0
    for entry in entries:
        if isinstance(entry, (list, tuple)):
            total += sum(1 for x in entry if str(x).strip())
        elif str(entry).strip():
            total += 1
    if total == 0:
        raise ValueError(f"{json_path}: no items")
    return total


def cycle_targets_for_count(
    n_items: int, cycle: Sequence[str] = CYCLE_TOKENS
) -> np.ndarray:
    """n items -> (4n,) target sequence of state indices, one cycle per item."""
    if n_items <= 0:
        raise ValueError(f"n_items must be positive, got {n_items}")
    idx: List[int] = []
    for tok in cycle:
        name = _token_to_state_name(tok)
        if name is None:
            raise ValueError(
                f"cycle token {tok!r} is not in TOKEN_TO_STATE; "
                f"update CYCLE_TOKENS in data.py to match states.py"
            )
        idx.append(STATE_TO_IDX[name])
    return np.asarray(idx * int(n_items), dtype=np.int64)


def compute_label_priors(samples: Sequence["PicklistSample"], smoothing: float = 1.0) -> np.ndarray:
    """Class frequencies over all dense frame labels, for HMM emission scaling.

    The network learns p(state | frame) under the training class balance.
    Dividing by this prior converts posteriors to scaled likelihoods, which
    stops frequent states from swallowing frames at segment boundaries.
    """
    counts = np.full(NUM_STATES, float(smoothing))
    for s in samples:
        if s.labels is None:
            continue
        valid = s.labels[s.labels >= 0]
        if valid.size:
            counts += np.bincount(valid, minlength=NUM_STATES)
    return counts / counts.sum()


def min_frames_for_targets(targets: Sequence[int]) -> int:
    """Shortest frame count that can emit `targets` under CTC."""
    targets = list(targets)
    if not targets:
        return 0
    repeats = sum(1 for a, b in zip(targets, targets[1:]) if a == b)
    return len(targets) + repeats


def discover_picklist_ids(
    feature_source: FeatureSource,
    features_dir: Path,
    labels_dir: Optional[Path] = None,
    require_labels: bool = True,
) -> List[str]:
    """Return sorted picklist ids that have features, optionally requiring labels."""
    feat_ids = set()
    if feature_source == "processed_3d":
        for p in features_dir.iterdir():
            if p.is_file() and not p.name.startswith("."):
                m = re.match(r"picklist_(\d+)$", p.name, re.I)
                if m:
                    feat_ids.add(m.group(1))
    else:
        for path in features_dir.glob("*_holding_timeseries.csv"):
            stem = path.name[: -len("_holding_timeseries.csv")]
            m = re.match(r"picklist_(\d+)$", stem, re.I)
            if m:
                feat_ids.add(m.group(1))

    out: List[str] = []
    for pid in sorted(feat_ids, key=int):
        if require_labels and labels_dir is not None:
            label_path = labels_dir / f"picklist_{pid}.csv"
            if not label_path.is_file():
                continue
            # parse_boundary_labels returns a 2-tuple, which is always truthy:
            # unpack before testing or empty label files pass through.
            boundaries, tokens = parse_boundary_labels(label_path)
            if not boundaries:
                continue
            if tokens_to_target_sequence(tokens).size == 0:
                continue
        out.append(pid)
    return out


def expand_boundary_labels(boundaries: List[Tuple[int, str]], n_frames: int) -> np.ndarray:
    """Expand frame-boundary tokens into dense per-frame state indices."""
    labels = np.full(n_frames, -1, dtype=np.int64)
    if not boundaries:
        return labels

    for i, (start_frame, token) in enumerate(boundaries):
        end_frame = boundaries[i + 1][0] if i + 1 < len(boundaries) else n_frames
        start_frame = max(0, min(start_frame, n_frames))
        end_frame = max(start_frame, min(end_frame, n_frames))
        state_name = _token_to_state_name(token)
        if state_name is None:
            continue
        labels[start_frame:end_frame] = STATE_TO_IDX[state_name]

    first_labeled = next((i for i in range(n_frames) if labels[i] >= 0), None)
    if first_labeled is not None and first_labeled > 0:
        labels[:first_labeled] = labels[first_labeled]
    prev = labels[first_labeled] if first_labeled is not None else STATE_TO_IDX["CARRY_EMPTY"]
    for t in range(n_frames):
        if labels[t] < 0:
            labels[t] = prev
        else:
            prev = labels[t]
    return labels


def load_frame_labels(label_path: Path, n_frames: int) -> np.ndarray:
    """Read any label file (hand-annotated or predicted) -> (T,) dense states."""
    boundaries, _ = parse_boundary_labels(label_path)
    return expand_boundary_labels(boundaries, n_frames)


def load_ground_truth_counts(path: Path) -> Dict[str, int]:
    """Parse ground_truth.csv: count non-empty object codes per picklist column."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    headers = rows[0]
    counts: Dict[str, int] = {}
    for j, header in enumerate(headers):
        m = re.match(r"picklist_(\d+)$", header.strip(), re.I)
        if not m:
            continue
        pid = m.group(1)
        n = sum(1 for r in rows[1:] if j < len(r) and r[j].strip())
        if n > 0:
            counts[pid] = n
    return counts


def load_processed_3d_features(path: Path) -> np.ndarray:
    """Load (T, 3) matrix: p_hold, direction, net_aruco."""
    rows: List[List[float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}: expected 3 floats, got {len(parts)}")
            rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: no feature rows")
    return np.asarray(rows, dtype=np.float64)


def _load_timeseries_csv(csv_path: Path, value_col: str) -> Dict[int, float]:
    by_frame: Dict[int, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or value_col not in reader.fieldnames:
            raise ValueError(f"{csv_path}: missing column {value_col!r}")
        for row in reader:
            raw = row[value_col].strip()
            by_frame[int(row["frame_idx"])] = float(raw) if raw else float("nan")
    if not by_frame:
        raise ValueError(f"{csv_path}: no data rows")
    return by_frame


def _forward_fill(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    if len(out) == 0:
        return out
    if np.isnan(out[0]):
        out[0] = 0.0
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    return out


def _expand_full_rate(
    knot_frames: np.ndarray,
    knot_values: Dict[str, np.ndarray],
    total_frames: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    knot_idx = knot_frames.astype(np.int64)
    for key, vals in knot_values.items():
        arr = np.empty(total_frames, dtype=np.float64)
        for i, start in enumerate(knot_idx):
            end = int(knot_idx[i + 1]) if i + 1 < len(knot_idx) else total_frames
            end = min(end, total_frames)
            arr[start:end] = vals[i]
        if knot_idx[0] > 0:
            arr[: knot_idx[0]] = vals[0]
        out[key] = arr
    return out


def load_csv_outputs_features(
    csv_dir: Path, stem: str, match_length: Optional[int] = None
) -> np.ndarray:
    """Load and expand raw detector CSVs to (T, 5)."""
    hold_path = csv_dir / f"{stem}_holding_timeseries.csv"
    dir_path = csv_dir / f"{stem}_hand_direction.csv"
    aruco_path = csv_dir / f"{stem}_aruco_net_timeseries.csv"

    p_hold_by = _load_timeseries_csv(hold_path, "p_hold")
    dir_by = _load_timeseries_csv(dir_path, "direction")

    ref_frames = np.array(sorted(p_hold_by.keys()), dtype=np.int64)
    p_hold_knots = np.array([p_hold_by[int(fi)] for fi in ref_frames], dtype=np.float64)
    dir_knots = _forward_fill(np.array([dir_by.get(int(fi), float("nan")) for fi in ref_frames]))

    aruco_net: List[float] = []
    pick_counts: List[float] = []
    place_counts: List[float] = []
    with aruco_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        aruco_by_frame = {int(row["frame_idx"]): row for row in reader}
    for fi in ref_frames:
        row = aruco_by_frame.get(int(fi), {})
        aruco_net.append(float(row.get("net_aruco", 0) or 0))
        pick_counts.append(float(row.get("pick_count", 0) or 0))
        place_counts.append(float(row.get("place_count", 0) or 0))

    total_frames = match_length if match_length is not None else int(ref_frames[-1]) + 1
    knots = {
        "p_hold": p_hold_knots,
        "direction": dir_knots,
        "net_aruco": np.array(aruco_net, dtype=np.float64),
        "pick_count": np.array(pick_counts, dtype=np.float64),
        "place_count": np.array(place_counts, dtype=np.float64),
    }
    full = _expand_full_rate(ref_frames, knots, total_frames)
    return np.column_stack(
        [full["p_hold"], full["direction"], full["net_aruco"], full["pick_count"], full["place_count"]]
    )


def load_picklist_sample(
    picklist_id: str,
    feature_source: FeatureSource,
    features_dir: Path,
    labels_dir: Optional[Path],
    ground_truth: Optional[Dict[str, int]] = None,
    fps: float = 30.0,
    drop_trailing_cycle_start: bool = True,
) -> PicklistSample:
    stem = f"picklist_{picklist_id}"

    if feature_source == "processed_3d":
        feat_path = features_dir / stem
        if not feat_path.is_file():
            feat_path = features_dir / picklist_id
        features = load_processed_3d_features(feat_path)
    else:
        match_len = None
        ref_3d = DEFAULT_FEATURES_3D / stem
        if ref_3d.is_file():
            with ref_3d.open(encoding="utf-8", errors="replace") as f:
                match_len = sum(1 for line in f if line.strip())
        features = load_csv_outputs_features(features_dir, stem, match_length=match_len)

    n_frames = features.shape[0]

    labels: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    if labels_dir is not None:
        label_path = labels_dir / f"{stem}.csv"
        if label_path.is_file():
            boundaries, tokens = parse_boundary_labels(label_path)
            if boundaries:
                labels = expand_boundary_labels(boundaries, n_frames)
                seq = tokens_to_target_sequence(
                    tokens, drop_trailing_cycle_start=drop_trailing_cycle_start
                )
                targets = seq if seq.size else None

    obj_count = ground_truth.get(picklist_id) if ground_truth else None
    return PicklistSample(
        picklist_id=picklist_id,
        features=features,
        labels=labels,
        targets=targets,
        fps=fps,
        object_count=obj_count,
    )


def load_dataset(
    picklist_ids: Sequence[str],
    feature_source: FeatureSource,
    features_dir: Path,
    labels_dir: Path,
    ground_truth_path: Optional[Path] = None,
    fps: float = 30.0,
    require_ctc_feasible: bool = False,
    require_frame_labels: bool = False,
    drop_trailing_cycle_start: bool = True,
) -> List[PicklistSample]:
    gt = load_ground_truth_counts(ground_truth_path) if ground_truth_path else {}
    samples: List[PicklistSample] = []
    for pid in picklist_ids:
        try:
            s = load_picklist_sample(
                pid, feature_source, features_dir, labels_dir, gt, fps,
                drop_trailing_cycle_start=drop_trailing_cycle_start,
            )
        except (ValueError, OSError, KeyError) as exc:
            print(f"Skipping picklist_{pid}: {exc}")
            continue
        if require_frame_labels:
            if s.labels is None or not (s.labels >= 0).any():
                print(f"Skipping picklist_{pid}: no dense frame labels")
                continue
            if s.labels.shape[0] != s.features.shape[0]:
                print(
                    f"Skipping picklist_{pid}: {s.labels.shape[0]} labels vs "
                    f"{s.features.shape[0]} feature frames"
                )
                continue
        if require_ctc_feasible:
            if s.targets is None:
                print(f"Skipping picklist_{pid}: no usable target sequence")
                continue
            need = min_frames_for_targets(s.targets)
            if s.features.shape[0] < need:
                print(
                    f"Skipping picklist_{pid}: {s.features.shape[0]} frames < "
                    f"{need} required for {len(s.targets)} labels"
                )
                continue
        samples.append(s)
    return samples


def compute_normalization_stats(samples: Sequence[PicklistSample]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([s.features for s in samples])
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def train_val_split(
    picklist_ids: List[str], val_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    ids = list(picklist_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def feature_dim_for_source(feature_source: FeatureSource) -> int:
    return 3 if feature_source == "processed_3d" else 5


def feature_names_for_source(feature_source: FeatureSource) -> Tuple[str, ...]:
    return FEATURE_NAMES_3D if feature_source == "processed_3d" else FEATURE_NAMES_CSV
