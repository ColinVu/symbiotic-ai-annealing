"""EAF → CSV Annotation Converter.

Converts ELAN Linguistic Annotator (EAF) annotation files to the CSV format
expected by the HTK HMM training pipeline.

EAF annotation-value → HMM state mapping
-----------------------------------------
The EAF files use fine-grained labels that embed both the action type and an
optional object/place qualifier:

    EAF value           HMM state
    ------------------  -----------
    carry_empty         CARRY_EMPTY
    pick_<anything>     PICK
    carry_<anything>    CARRY_WITH   (i.e. anything that isn't carry_empty)
    place_<anything>    PLACE

Times are stored in the EAF as integer milliseconds and are converted to
floating-point seconds in the output CSV.

Output CSV columns: timestamp_start, timestamp_end, state

Usage (from symbiotic-ai/)::

    # Convert a single file (output alongside the input)
    python -m symbiote.eaf_to_csv path/to/file.eaf

    # Specify output path explicitly
    python -m symbiote.eaf_to_csv path/to/file.eaf --output path/to/out.csv

    # Convert every .eaf in a directory, save CSVs to another directory
    python -m symbiote.eaf_to_csv --input-dir hmm-testing/eaf_labels \\
                                   --output-dir hmm-testing/picklist_labels

    # Convert any .eaf files already sitting in picklist_labels/
    python -m symbiote.eaf_to_csv --input-dir hmm-testing/picklist_labels \\
                                   --output-dir hmm-testing/picklist_labels
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# State mapping
# ---------------------------------------------------------------------------

def _map_eaf_value_to_state(value: str) -> Optional[str]:
    """Map a raw EAF annotation value to an HMM state label.

    Returns ``None`` if the value cannot be mapped (unknown label).
    """
    v = value.strip().lower()

    if v == "carry_empty":
        return "CARRY_EMPTY"
    if v.startswith("pick_"):
        return "PICK"
    if v.startswith("carry_"):
        # catch-all: any carry_<colour> etc. that is not carry_empty
        return "CARRY_WITH"
    if v.startswith("place_"):
        return "PLACE"

    return None


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_eaf(eaf_path: str) -> pd.DataFrame:
    """Parse an EAF file and return a sorted annotation DataFrame.

    Columns: ``timestamp_start`` (seconds), ``timestamp_end`` (seconds),
    ``state`` (HMM state label).

    Raises:
        FileNotFoundError: If *eaf_path* does not exist.
        ValueError: If the file contains no parseable annotations.
    """
    eaf_path = str(eaf_path)
    if not os.path.isfile(eaf_path):
        raise FileNotFoundError(f"EAF file not found: {eaf_path}")

    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # ------------------------------------------------------------------
    # 1. Build time-slot ID → millisecond value map
    # ------------------------------------------------------------------
    time_slots: Dict[str, int] = {}
    time_order = root.find("TIME_ORDER")
    if time_order is not None:
        for ts in time_order.findall("TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_val = ts.get("TIME_VALUE")
            if ts_id and ts_val is not None:
                time_slots[ts_id] = int(ts_val)

    if not time_slots:
        raise ValueError(f"No TIME_SLOT entries found in {eaf_path}")

    # ------------------------------------------------------------------
    # 2. Parse ALIGNABLE_ANNOTATIONs from all tiers
    # ------------------------------------------------------------------
    rows: List[dict] = []
    skipped: List[Tuple[str, str]] = []

    for tier in root.findall("TIER"):
        for annotation in tier.findall("ANNOTATION"):
            aa = annotation.find("ALIGNABLE_ANNOTATION")
            if aa is None:
                continue

            ref1 = aa.get("TIME_SLOT_REF1")
            ref2 = aa.get("TIME_SLOT_REF2")
            val_el = aa.find("ANNOTATION_VALUE")
            if val_el is None or ref1 is None or ref2 is None:
                continue

            raw_value = (val_el.text or "").strip()
            state = _map_eaf_value_to_state(raw_value)

            if state is None:
                skipped.append((raw_value, aa.get("ANNOTATION_ID", "?")))
                continue

            start_ms = time_slots.get(ref1)
            end_ms   = time_slots.get(ref2)
            if start_ms is None or end_ms is None:
                continue

            rows.append({
                "timestamp_start": start_ms / 1000.0,
                "timestamp_end":   end_ms   / 1000.0,
                "state":           state,
            })

    if skipped:
        print(
            f"[eaf_to_csv] WARNING: {len(skipped)} annotation(s) in "
            f"'{os.path.basename(eaf_path)}' could not be mapped to an HMM "
            f"state and were skipped:"
        )
        for raw, ann_id in skipped:
            print(f"  annotation {ann_id}: '{raw}'")

    if not rows:
        raise ValueError(
            f"No mappable annotations found in {eaf_path}. "
            "Expected values starting with pick_, carry_, or place_, "
            "or exactly 'carry_empty'."
        )

    df = pd.DataFrame(rows).sort_values("timestamp_start").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# File-level conversion
# ---------------------------------------------------------------------------

def convert_file(
    eaf_path: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Convert a single EAF file to a CSV annotation file.

    Args:
        eaf_path:   Path to the source EAF file.
        output_csv: Destination CSV path.  If ``None``, the CSV is placed
                    alongside the EAF file with the same stem.

    Returns:
        Absolute path to the written CSV file.
    """
    if output_csv is None:
        output_csv = str(Path(eaf_path).with_suffix(".csv"))

    df = parse_eaf(eaf_path)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    if verbose:
        print(
            f"[eaf_to_csv] {os.path.basename(eaf_path)} → "
            f"{output_csv}  ({len(df)} annotations)"
        )
    return output_csv


def convert_directory(
    input_dir: str,
    output_dir: str,
    verbose: bool = True,
) -> List[str]:
    """Convert every ``.eaf`` file in *input_dir* and save CSVs to *output_dir*.

    Returns:
        List of paths to the written CSV files.
    """
    input_dir  = str(input_dir)
    output_dir = str(output_dir)

    eaf_files = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() == ".eaf"
    )

    if not eaf_files:
        if verbose:
            print(f"[eaf_to_csv] No .eaf files found in {input_dir}")
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    for eaf_path in eaf_files:
        out_csv = os.path.join(output_dir, eaf_path.stem + ".csv")
        try:
            written.append(convert_file(str(eaf_path), out_csv, verbose=verbose))
        except Exception as exc:
            print(f"[eaf_to_csv] ERROR converting {eaf_path.name}: {exc}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.eaf_to_csv",
        description="Convert ELAN .eaf annotation files to HMM training CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mutually exclusive: single file vs. directory
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "eaf_file",
        nargs="?",
        metavar="EAF_FILE",
        help="Single .eaf file to convert",
    )
    mode.add_argument(
        "--input-dir",
        metavar="DIR",
        help="Directory of .eaf files to convert in batch",
    )

    p.add_argument(
        "--output",
        metavar="CSV",
        help="Output CSV path (single-file mode only)",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default=str(_ROOT / "hmm-testing" / "picklist_labels"),
        help=(
            "Output directory for batch mode "
            "(default: hmm-testing/picklist_labels)"
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    verbose = not args.quiet

    if args.eaf_file:
        # Single-file mode
        try:
            convert_file(args.eaf_file, output_csv=args.output, verbose=verbose)
        except Exception as exc:
            print(f"[eaf_to_csv] ERROR: {exc}")
            sys.exit(1)

    elif args.input_dir:
        # Batch / directory mode
        written = convert_directory(args.input_dir, args.output_dir, verbose=verbose)
        if verbose and written:
            print(f"[eaf_to_csv] Converted {len(written)} file(s) to {args.output_dir}")

    else:
        # Default batch: convert any .eaf files in picklist_labels/
        default_dir = str(_ROOT / "hmm-testing" / "picklist_labels")
        written = convert_directory(default_dir, default_dir, verbose=verbose)
        if verbose and written:
            print(f"[eaf_to_csv] Converted {len(written)} file(s) in {default_dir}")
        elif not written and verbose:
            print(
                "[eaf_to_csv] No .eaf files found. "
                "Pass an EAF_FILE argument or --input-dir to specify a source."
            )


if __name__ == "__main__":
    main()
