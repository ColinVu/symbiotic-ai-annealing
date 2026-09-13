"""Bulk HMM forced alignment over processed_features_3d -> picklist_labels CSVs.

Writes compact frame_index,code files to hmm-testing/picklist_labels by default,
the same format annealing already consumes.

    python -m nn_model.bulk_predict \\
        --checkpoint nn_model/checkpoints/best_model.pt \\
        --min-duration 8 --prior-scale 0.5 --trailing-m --report-spans
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .data import (
    CYCLE_TOKENS,
    DEFAULT_FEATURES_3D,
    DEFAULT_GROUND_TRUTH,
    DEFAULT_LABELS,
    DEFAULT_PICKLIST_JSONS,
    DEFAULT_PICKLIST_LABELS,
    NUM_STATES,
    load_ground_truth_counts,
    load_processed_3d_features,
)
from .decode import decode
from .hmm_align import segments_from_frames, viterbi_align
from .labels_io import write_label_file
from .predict import load_checkpoint, predict_logits
from .sequences import (
    check_cycle_structure,
    get_sequence,
    load_label_sequences,
    normalize_picklist_id,
)
from .states import IDX_TO_STATE, STATE_TO_TOKEN


def discover_processed_3d_picklists(features_dir: Path) -> List[str]:
    """Sorted picklist ids from processed_features_3d/picklist_* files.

    Note the `$` anchor: these files have no extension, so picklist_007.csv
    is not discovered.
    """
    ids: List[str] = []
    for path in features_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        m = re.match(r"picklist_(\d+)$", path.name, re.I)
        if m:
            ids.append(m.group(1))
    return sorted(ids, key=int)


def frame_labels_to_boundaries(frame_labels: np.ndarray) -> List[Tuple[int, str]]:
    """Dense per-frame state indices -> (frame_index, code) transition rows."""
    if len(frame_labels) == 0:
        return []
    rows: List[Tuple[int, str]] = [(0, STATE_TO_TOKEN[IDX_TO_STATE[int(frame_labels[0])]])]
    prev_idx = int(frame_labels[0])
    for t in range(1, len(frame_labels)):
        cur_idx = int(frame_labels[t])
        if cur_idx != prev_idx:
            rows.append((t, STATE_TO_TOKEN[IDX_TO_STATE[cur_idx]]))
            prev_idx = cur_idx
    return rows


def check_checkpoint_head(logits_dim: int, config: dict) -> None:
    """Refuse anything that is not a frame-level (cross-entropy) head."""
    loss = str(config.get("loss", "")).lower()
    if loss == "ctc" or logits_dim == NUM_STATES + 1:
        raise SystemExit(
            f"This checkpoint has a CTC head (C={logits_dim}, loss={loss!r}). "
            f"HMM forced alignment needs a cross-entropy checkpoint with "
            f"{NUM_STATES} outputs and no blank column. Retrain with "
            f"nn_model.train, or use the original bulk_predict for CTC."
        )
    if logits_dim != NUM_STATES:
        raise SystemExit(
            f"Model emits {logits_dim} classes but there are {NUM_STATES} states."
        )
    print(f"Frame classifier head: C={logits_dim} (cross-entropy)")


def report_sequence_problems(
    target_seqs: Dict[str, np.ndarray],
    origin: Dict[str, str],
    gt_counts: Dict[str, int],
) -> None:
    """Warn about targets that are not clean maei cycles, before aligning.

    Forced alignment reproduces its target exactly, so a malformed sequence
    produces a confident alignment that distributes the wrong number of cycles
    across the video. Nothing downstream will notice.
    """
    per_cycle = len(CYCLE_TOKENS)
    bad = []
    for pid, seq in sorted(target_seqs.items(), key=lambda kv: int(kv[0])):
        ok, why = check_cycle_structure(seq)
        if not ok:
            bad.append((pid, origin.get(pid, "?"), why))

    if bad:
        print(f"\nWARNING: {len(bad)} target sequence(s) are not clean "
              f"{''.join(CYCLE_TOKENS)} cycles:")
        for pid, src, why in bad[:20]:
            print(f"  picklist_{pid} [{src}]: {why}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")
        print("  Run `python -m nn_model.check_sequences` for detail.")

    if gt_counts:
        mismatched = [
            (pid, len(seq) // per_cycle, gt_counts[pid])
            for pid, seq in sorted(target_seqs.items(), key=lambda kv: int(kv[0]))
            if pid in gt_counts and len(seq) // per_cycle != gt_counts[pid]
        ]
        if mismatched:
            print(f"\nWARNING: {len(mismatched)} sequence(s) disagree with "
                  f"ground_truth.csv on item count:")
            for pid, n_seq, n_gt in mismatched[:20]:
                print(f"  picklist_{pid}: sequence has {n_seq} cycles, ground truth has {n_gt}")
            if len(mismatched) > 20:
                print(f"  ... and {len(mismatched) - 20} more")


def generate_label_csvs(
    checkpoint: Union[str, Path],
    features_dir: Union[str, Path],
    output_dir: Union[str, Path],
    picklist_ids: Sequence[str],
    *,
    picklist_jsons: Optional[Union[str, Path]] = None,
    labels_dir: Optional[Union[str, Path]] = None,
    ground_truth: Optional[Union[str, Path]] = None,
    sequence_source: str = "json",
    trailing_m: bool = False,
    whole_cycles: bool = False,
    skip_malformed: bool = False,
    fallback_decode: str = "constrained",
    output_format: str = "csv",
    device: str = "cpu",
    prior_scale: float = 1.0,
    min_duration: int = 1,
    transition_scale: float = 0.0,
    report_spans: bool = False,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Align one checkpoint over explicit picklist IDs and write compact CSVs.

    Returns a mapping of numeric picklist id -> dense per-frame state indices.
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    ids = [normalize_picklist_id(str(pid)) for pid in picklist_ids]
    if not ids:
        raise SystemExit("No picklist ids given to generate_label_csvs")

    json_dir = Path(picklist_jsons) if picklist_jsons is not None else None
    target_seqs: Dict[str, np.ndarray] = {}
    origin: Dict[str, str] = {}
    if json_dir is not None or labels_dir is not None:
        target_seqs, origin = load_label_sequences(
            labels_dir=labels_dir if labels_dir is not None else DEFAULT_LABELS,
            picklist_jsons=json_dir if json_dir is not None else DEFAULT_PICKLIST_JSONS,
            source=sequence_source,
            trailing_m=trailing_m,
            whole_cycles=whole_cycles,
        )

    gt_counts: Dict[str, int] = {}
    if ground_truth is not None and Path(ground_truth).is_file():
        gt_counts = load_ground_truth_counts(Path(ground_truth))
        report_sequence_problems(target_seqs, origin, gt_counts)

    torch_device = torch.device(device)
    model, config, mean, std = load_checkpoint(Path(checkpoint), torch_device)
    if verbose and config.get("feature_source") != "processed_3d" and config.get("input_dim") != 3:
        print(
            f"Warning: checkpoint feature_source={config.get('feature_source')!r} "
            f"input_dim={config.get('input_dim')}; this function always reads "
            f"processed_features_3d (3-D)."
        )

    self_loop_logprobs = None
    if config.get("self_loop_logprobs") is not None:
        self_loop_logprobs = np.asarray(config["self_loop_logprobs"], dtype=np.float64)

    priors = None
    if config.get("priors") is not None:
        priors = np.asarray(config["priors"], dtype=np.float64)

    output_dir.mkdir(parents=True, exist_ok=True)
    head_checked = False
    dense: Dict[str, np.ndarray] = {}
    span_lengths: List[int] = []
    used_src: Counter = Counter()
    skipped = 0

    for pid in ids:
        try:
            features = load_processed_3d_features(features_dir / f"picklist_{pid}")
        except (ValueError, OSError) as exc:
            if verbose:
                print(f"Skipping picklist_{pid}: {exc}")
            skipped += 1
            continue

        logits = predict_logits(model, features, mean, std, torch_device)
        if not head_checked:
            check_checkpoint_head(logits.shape[1], config)
            head_checked = True

        targets = get_sequence(target_seqs, pid)
        src = origin.get(pid, "?")

        reason = None
        if targets is None or len(targets) == 0:
            reason = "no target sequence"
        else:
            need = len(targets) * int(min_duration)
            if logits.shape[0] < need:
                reason = (
                    f"{logits.shape[0]} frames < {need} required for "
                    f"{len(targets)} labels at min_duration={min_duration}"
                )
            elif skip_malformed and not check_cycle_structure(targets)[0]:
                reason = f"malformed target sequence ({check_cycle_structure(targets)[1]})"

        if reason is None:
            path, score = viterbi_align(
                logits,
                [int(t) for t in targets],
                priors=priors,
                prior_scale=prior_scale,
                min_duration=int(min_duration),
                self_loop_logprobs=self_loop_logprobs,
                transition_scale=transition_scale,
            )
            used_src[src] += 1
            if verbose:
                msg = (
                    f"picklist_{pid}: aligned {len(targets)} labels [{src}] over "
                    f"{logits.shape[0]} frames (score={score:.1f})"
                )
                if report_spans:
                    widths = [b - a for a, b, _ in segments_from_frames(path)]
                    span_lengths.extend(widths)
                    msg += f", median segment {int(np.median(widths))} frame(s)"
                print(msg)
        else:
            if verbose:
                print(f"picklist_{pid}: {reason}; falling back to {fallback_decode}")
            if fallback_decode == "skip":
                skipped += 1
                continue
            path = decode(logits, mode="constrained")

        write_label_file(
            output_dir / f"picklist_{pid}.csv",
            frame_labels_to_boundaries(path),
            fmt=output_format,
        )
        dense[pid] = np.asarray(path, dtype=np.int64)

    if verbose:
        if used_src:
            print("\nAligned from: " + ", ".join(f"{v} {k}" for k, v in sorted(used_src.items())))
        if span_lengths:
            print(f"Median segment length across corpus: {int(np.median(span_lengths))} frame(s)")
        print(
            f"Wrote {len(dense)} label CSV(s) -> {output_dir}"
            + (f" ({skipped} skipped)" if skipped else "")
        )
    return dense


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HMM forced alignment over all processed_features_3d files.",
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_3D)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PICKLIST_LABELS,
        help="Compact frame_index,code CSVs for annealing (default: hmm-testing/picklist_labels).",
    )
    ap.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    ap.add_argument(
        "--output-format",
        choices=["htk", "csv"],
        default="csv",
        help="csv writes frame_index,code (annealing picklist_labels). htk writes quoted tab rows.",
    )
    ap.add_argument("--device", default="cpu")

    g = ap.add_argument_group("target sequences")
    g.add_argument(
        "--labels-dir",
        type=Path,
        default=DEFAULT_LABELS,
        help="csv_labels directory of hand annotations.",
    )
    g.add_argument(
        "--picklist-jsons",
        type=Path,
        default=None,
        help="Directory of picklist_NNN.json item lists (default: "
        "hmm-testing/picklist_jsons). May also be a single .json mapping file.",
    )
    g.add_argument(
        "--sequence-source",
        choices=["auto", "labels", "json"],
        default="json",
        help="json (default): every sequence is built from picklist_jsons as "
        "n items -> n x maei, so it is a clean multiple of 4 by construction. "
        "auto: prefer hand annotations where they exist, which means a typo'd "
        "or missing token in an annotation silently produces a malformed "
        "target. labels: annotations only.",
    )
    g.add_argument(
        "--whole-cycles",
        action="store_true",
        help="Crop every target to a whole number of complete maei cycles, "
        "dropping a partial cycle at either end. An annotation that opens mid "
        "cycle, or that marks the operator reaching for one last item as the "
        "video ends, otherwise forces the aligner through a partial cycle. "
        "No-op for JSON-derived sequences.",
    )
    g.add_argument(
        "--trailing-m",
        action="store_true",
        help="Append a final `m` to every target sequence (n x maei + m). The "
        "video ends with the operator empty handed; without this the aligner "
        "must stretch the last `i` to the end of the video and that boundary "
        "lands late. Applied to both sources, so annotated and JSON-only "
        "picklists get the same target shape.",
    )
    g.add_argument(
        "--skip-malformed",
        action="store_true",
        help="Skip picklists whose target sequence is not clean maei cycles, "
        "instead of aligning against it and warning.",
    )
    g.add_argument(
        "--fallback-decode",
        choices=["constrained", "skip"],
        default="skip",
        help="What to do when a picklist has no usable target sequence. "
        "'constrained' runs cycle Viterbi, which keeps the label order but "
        "guesses the number of cycles.",
    )
    g.add_argument("--report-spans", action="store_true")

    h = ap.add_argument_group("alignment")
    h.add_argument("--prior-scale", type=float, default=1.0,
                   help="Exponent on the class prior. 0 disables prior division.")
    h.add_argument("--min-duration", type=int, default=1,
                   help="Minimum frames per label. Raise to kill short segments.")
    h.add_argument("--transition-scale", type=float, default=0.0,
                   help="Multiplier on the ML transition log-probs from the "
                        "checkpoint. 1.0 is the correctly normalised model. "
                        "Measured effect is negligible at 1.0 and degrades "
                        "alignment above it -- use --min-duration to control "
                        "segment length. 0 disables transitions.")
    args = ap.parse_args()

    features_dir = args.features_dir.resolve()
    if not features_dir.is_dir():
        raise SystemExit(f"Features directory not found: {features_dir}")

    picklist_ids = discover_processed_3d_picklists(features_dir)
    if not picklist_ids:
        raise SystemExit(f"No picklist_* files in {features_dir}")

    json_dir = args.picklist_jsons if args.picklist_jsons is not None else DEFAULT_PICKLIST_JSONS
    print(f"Reading item counts from: {json_dir}")
    generate_label_csvs(
        checkpoint=args.checkpoint,
        features_dir=features_dir,
        output_dir=args.output_dir,
        picklist_ids=picklist_ids,
        picklist_jsons=json_dir,
        labels_dir=args.labels_dir,
        ground_truth=args.ground_truth,
        sequence_source=args.sequence_source,
        trailing_m=args.trailing_m,
        whole_cycles=args.whole_cycles,
        skip_malformed=args.skip_malformed,
        fallback_decode=args.fallback_decode,
        output_format=args.output_format,
        device=args.device,
        prior_scale=args.prior_scale,
        min_duration=args.min_duration,
        transition_scale=args.transition_scale,
        report_spans=args.report_spans,
        verbose=True,
    )


if __name__ == "__main__":
    main()