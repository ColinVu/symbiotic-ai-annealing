"""Load per-picklist target sequences for forced alignment.

Three sources, in decreasing order of how much they actually know:

  labels  csv_labels/picklist_NNN.csv - hand annotations, parsed by
          data.parse_boundary_labels. The real observed sequence.
  json    picklist_jsons/picklist_NNN.json - item lists. Each item implies one
          maei cycle, so n items -> n x maei. Covers picklists with no hand
          annotation, which is most of them.
  json    supplied directly as a {"136": ["m","a",...]} mapping.

Both loaders return {picklist_id: np.ndarray of state indices}.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import (
    CYCLE_TOKENS,
    _token_to_state_name,
    DEFAULT_LABELS,
    DEFAULT_PICKLIST_JSONS,
    count_picklist_items,
    cycle_targets_for_count,
    parse_boundary_labels,
    tokens_to_target_sequence,
)
from .states import IDX_TO_STATE, STATE_TO_IDX, STATE_TO_TOKEN

SequenceMap = Dict[str, np.ndarray]


def normalize_picklist_id(raw: str) -> str:
    """'picklist_011' / '011' / 'picklist_011.json' -> '011'."""
    m = re.search(r"(\d+)", str(raw).strip())
    return m.group(1) if m else str(raw).strip()


def get_sequence(seqs: SequenceMap, pid: str) -> Optional[np.ndarray]:
    """Look up a picklist id tolerant of zero-padding differences.

    Feature files may be picklist_11 while the JSON is picklist_011.
    """
    if pid in seqs:
        return seqs[pid]
    digits = normalize_picklist_id(pid)
    for cand in (digits, digits.lstrip("0") or "0", digits.zfill(3), digits.zfill(2)):
        if cand in seqs:
            return seqs[cand]
    return None


def load_sequences_from_labels_dir(
    labels_dir: Path = DEFAULT_LABELS,
    drop_trailing_cycle_start: bool = True,
) -> SequenceMap:
    """Read every csv_labels/picklist_*.csv -> target sequences."""
    labels_dir = Path(labels_dir)
    out: SequenceMap = {}
    if not labels_dir.is_dir():
        return out
    for path in sorted(labels_dir.glob("picklist_*.csv")):
        try:
            boundaries, tokens = parse_boundary_labels(path)
        except (ValueError, OSError) as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        if not boundaries:
            continue
        seq = tokens_to_target_sequence(
            tokens, drop_trailing_cycle_start=drop_trailing_cycle_start
        )
        if seq.size:
            out[normalize_picklist_id(path.stem)] = seq
    return out


def load_sequences_from_picklist_jsons(
    json_dir: Path = DEFAULT_PICKLIST_JSONS,
) -> Tuple[SequenceMap, Dict[str, int]]:
    """Read picklist_jsons/picklist_*.json -> (target sequences, item counts).

    Each item in the picklist requires one pass through CYCLE_TOKENS, so a
    picklist of n items becomes n repetitions of that cycle. No trailing
    cycle-start is appended: the JSON describes n clean cycles, which is
    exactly what the cropped training targets look like.
    """
    json_dir = Path(json_dir)
    out: SequenceMap = {}
    counts: Dict[str, int] = {}
    if not json_dir.is_dir():
        return out, counts
    for path in sorted(json_dir.glob("picklist_*.json")):
        try:
            n_items = count_picklist_items(path)
            seq = cycle_targets_for_count(n_items)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        pid = normalize_picklist_id(path.stem)
        out[pid] = seq
        counts[pid] = n_items
    return out, counts


def load_sequences_from_json_file(path: Path) -> SequenceMap:
    """Read {"136": ["m","a",...]} or {"136": "maei"} -> target sequences."""
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    out: SequenceMap = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            tokens: List[str] = [str(v).strip().lower() for v in value]
        else:
            s = str(value).strip().lower()
            parts = [p for p in re.split(r"[\s,;|]+", s) if p]
            tokens = parts if len(parts) > 1 else list(s)
        seq = tokens_to_target_sequence(tokens, drop_trailing_cycle_start=False)
        if seq.size:
            out[normalize_picklist_id(key)] = seq
    return out


def append_trailing_cycle_start(seq: np.ndarray) -> np.ndarray:
    """Append one trailing cycle-start (`m`) unless the sequence already ends in one.

    The picklist JSON describes n clean maei cycles, and annotations are cropped
    to match. But the video does not end the instant the last item is placed --
    the operator finishes empty handed, and those frames are genuinely `m`.
    Without a final `m` in the target, forced alignment has nowhere to put them
    and must extend the last `i` to the end of the video, which drags that
    boundary late.
    """
    start = STATE_TO_IDX[_token_to_state_name(CYCLE_TOKENS[0])]
    seq = np.asarray(seq, dtype=np.int64)
    if seq.size and int(seq[-1]) == start:
        return seq
    return np.concatenate([seq, np.asarray([start], dtype=np.int64)])


def describe_sequence(seq: Sequence[int]) -> str:
    """State indices -> token string, e.g. 'maeimaeim'."""
    return "".join(STATE_TO_TOKEN[IDX_TO_STATE[int(i)]] for i in seq)


def check_cycle_structure(seq: Sequence[int]) -> Tuple[bool, str]:
    """Is this sequence n x maei, optionally plus one trailing cycle-start?

    -> (ok, description). Nothing in tokens_to_target_sequence validates this:
    it drops sil and any unrecognised token silently, so a typo'd code in an
    annotation removes a label without warning. Losing an `m` merges two cycles
    into a seven-label run and the item count silently goes wrong.
    """
    toks = describe_sequence(seq)
    cycle = "".join(CYCLE_TOKENS)
    body, tail = toks, ""
    if len(toks) % len(cycle) == 1 and toks.endswith(CYCLE_TOKENS[0]):
        body, tail = toks[:-1], CYCLE_TOKENS[0]
    if len(body) % len(cycle):
        return False, f"length {len(toks)} is not a multiple of {len(cycle)}"
    if body != cycle * (len(body) // len(cycle)):
        for i, ch in enumerate(body):
            if ch != cycle[i % len(cycle)]:
                return False, f"breaks the cycle at position {i}: expected {cycle[i % len(cycle)]!r}, got {ch!r}"
    return True, f"{len(body) // len(cycle)} cycles" + (" + trailing m" if tail else "")


def crop_to_whole_cycles(seq: Sequence[int]) -> Tuple[np.ndarray, int, int]:
    """Trim a sequence to a whole number of complete maei cycles.

    -> (cropped, n_dropped_head, n_dropped_tail)

    Two things get dropped, both of which otherwise force the aligner through
    a partial cycle:

    head  labels before the first cycle-start. An annotation that opens mid
          cycle -- the video begins with the operator already carrying -- gives
          a target starting `e i m a e i ...`, and forced alignment reproduces
          that opening `e` faithfully.
    tail  an incomplete final cycle. Annotators often mark the operator
          reaching for one more item as the video ends, giving a target that
          finishes `... i m a`. Those two labels describe a pick that never
          completes.

    Anything left after cropping is guaranteed to be exactly n x maei, so every
    item in the output passes through all four states.
    """
    cycle = [STATE_TO_IDX[_token_to_state_name(t)] for t in CYCLE_TOKENS]
    idx = [int(v) for v in seq]

    head = 0
    while head < len(idx) and idx[head] != cycle[0]:
        head += 1

    body = idx[head:]
    kept = 0
    for j, v in enumerate(body):
        if v != cycle[j % len(cycle)]:
            break
        kept = j + 1
    kept -= kept % len(cycle)   # back off to the last complete cycle

    return np.asarray(body[:kept], dtype=np.int64), head, len(body) - kept


def enforce_whole_cycles(
    seqs: SequenceMap, origin: Dict[str, str], verbose: bool = True
) -> Tuple[SequenceMap, List[str]]:
    """Apply crop_to_whole_cycles across a sequence map. -> (seqs, dropped_ids).

    Picklists left with nothing at all are removed and returned, since a
    sequence of zero labels cannot be aligned.
    """
    out: SequenceMap = {}
    empty: List[str] = []
    changed = 0
    for pid, seq in seqs.items():
        cropped, head, tail = crop_to_whole_cycles(seq)
        if cropped.size == 0:
            empty.append(pid)
            continue
        if head or tail:
            changed += 1
            if verbose:
                bits = []
                if head:
                    bits.append(f"{head} leading")
                if tail:
                    bits.append(f"{tail} trailing")
                print(
                    f"  picklist_{pid} [{origin.get(pid, '?')}]: dropped "
                    f"{' and '.join(bits)} label(s) -> {cropped.size // len(CYCLE_TOKENS)} cycles"
                )
        out[pid] = cropped
    if verbose and (changed or empty):
        print(f"  cropped {changed} sequence(s) to whole cycles"
              + (f"; {len(empty)} left empty and dropped" if empty else ""))
    return out, empty


def load_label_sequences(
    labels_dir: Optional[Path] = None,
    picklist_jsons: Optional[Path] = None,
    source: str = "auto",
    trailing_m: bool = False,
    whole_cycles: bool = False,
) -> Tuple[SequenceMap, Dict[str, str]]:
    """Assemble target sequences from the requested sources.

    source:
        "auto"   hand annotations where they exist, JSON-derived cycles elsewhere
        "labels" csv_labels only
        "json"   picklist_jsons only (or a .json mapping file)

    whole_cycles: crop each sequence to a whole number of complete maei
    cycles, dropping a partial cycle at either end. Guarantees every item in
    the target passes through all four states. No-op for JSON-derived
    sequences, which are built that way.

    trailing_m: append one final cycle-start to every sequence, so the target
    reads n x maei + m. Both sources are normalised the same way -- annotations
    are cropped to n clean cycles first, then the `m` is added back -- so a
    picklist's target does not depend on whether it happens to be annotated.

    Returns (sequences, origin) where origin maps picklist id -> "labels"|"json",
    so callers can report which picklists were aligned against real
    annotations versus an assumed n x maei structure.
    """
    seqs: SequenceMap = {}
    origin: Dict[str, str] = {}

    json_src = Path(picklist_jsons) if picklist_jsons is not None else DEFAULT_PICKLIST_JSONS

    if source in ("auto", "json"):
        if json_src.is_file() and json_src.suffix.lower() == ".json":
            json_seqs = load_sequences_from_json_file(json_src)
        else:
            json_seqs, _ = load_sequences_from_picklist_jsons(json_src)
        for pid, seq in json_seqs.items():
            seqs[pid] = seq
            origin[pid] = "json"

    if source in ("auto", "labels"):
        label_seqs = load_sequences_from_labels_dir(
            Path(labels_dir) if labels_dir is not None else DEFAULT_LABELS
        )
        # Hand annotations win over the assumed cycle structure.
        for pid, seq in label_seqs.items():
            seqs[pid] = seq
            origin[pid] = "labels"

    if whole_cycles:
        seqs, empty = enforce_whole_cycles(seqs, origin)
        for pid in empty:
            origin.pop(pid, None)

    if trailing_m:
        seqs = {pid: append_trailing_cycle_start(seq) for pid, seq in seqs.items()}

    if not seqs:
        raise ValueError(
            f"no target sequences found (labels_dir={labels_dir}, picklist_jsons={json_src})"
        )
    return seqs, origin


def describe_cycle() -> str:
    return "".join(CYCLE_TOKENS)