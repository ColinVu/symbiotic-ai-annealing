from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .hmm_model import STATE_LABELS, STATE_SYMBOLS


@dataclass
class AnnotationSegment:
    start_time_ms: int
    end_time_ms: int
    label: str


# Mapping from EAF annotation values to state symbols
# EAF labels can be like "pick_red", "carry_red", "place_1", "carry_empty", etc.
# We need to map them to: a (pick), e (carry), i (place), m (carry_empty)
EAF_TO_STATE: Dict[str, str] = {
    # Pick variations
    "pick": "a",
    "pick_red": "a",
    "pick_green": "a",
    "pick_blue": "a",
    # Carry variations
    "carry": "e",
    "carry_red": "e",
    "carry_green": "e",
    "carry_blue": "e",
    # Place variations
    "place": "i",
    "place_1": "i",
    "place_2": "i",
    "place_3": "i",
    "place_green": "i",
    # Carry empty
    "carry_empty": "m",
}


def parse_eaf(eaf_path: Path) -> List[AnnotationSegment]:
    """Parse an ELAN annotation file (EAF) and extract time-aligned annotations.
    
    Args:
        eaf_path: Path to the .eaf file
        
    Returns:
        List of annotation segments with start/end times in milliseconds and labels
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    
    # Parse time slots
    time_slots: Dict[str, int] = {}
    time_order = root.find("TIME_ORDER")
    if time_order is not None:
        for time_slot in time_order.findall("TIME_SLOT"):
            slot_id = time_slot.get("TIME_SLOT_ID")
            time_value = time_slot.get("TIME_VALUE")
            if slot_id and time_value:
                time_slots[slot_id] = int(time_value)
    
    # Parse annotations
    segments: List[AnnotationSegment] = []
    tier = root.find(".//TIER")
    if tier is not None:
        for annotation in tier.findall(".//ALIGNABLE_ANNOTATION"):
            start_ref = annotation.get("TIME_SLOT_REF1")
            end_ref = annotation.get("TIME_SLOT_REF2")
            value_elem = annotation.find("ANNOTATION_VALUE")
            
            if start_ref and end_ref and value_elem is not None:
                start_time = time_slots.get(start_ref)
                end_time = time_slots.get(end_ref)
                label = value_elem.text
                
                if start_time is not None and end_time is not None and label:
                    segments.append(
                        AnnotationSegment(
                            start_time_ms=start_time,
                            end_time_ms=end_time,
                            label=label.strip(),
                        )
                    )
    
    # Sort by start time
    segments.sort(key=lambda s: s.start_time_ms)
    return segments


def eaf_to_state_sequence(
    eaf_path: Path,
    timestamps: List[float],
) -> List[str]:
    """Convert EAF annotations to a state sequence aligned with frame timestamps.
    
    Args:
        eaf_path: Path to the .eaf annotation file
        timestamps: List of frame timestamps in seconds
        
    Returns:
        List of state symbols (a, e, i, m) aligned with timestamps
    """
    segments = parse_eaf(eaf_path)
    
    if not segments:
        raise ValueError(f"No annotations found in {eaf_path}")
    
    # Convert timestamps to milliseconds for comparison
    timestamps_ms = [int(t * 1000) for t in timestamps]
    
    state_sequence: List[str] = []
    segment_idx = 0
    
    for ts_ms in timestamps_ms:
        # Find the segment that contains this timestamp
        while segment_idx < len(segments) - 1 and ts_ms >= segments[segment_idx].end_time_ms:
            segment_idx += 1
        
        if segment_idx < len(segments):
            segment = segments[segment_idx]
            if segment.start_time_ms <= ts_ms < segment.end_time_ms:
                # Map EAF label to state symbol
                state_symbol = EAF_TO_STATE.get(segment.label.lower())
                if state_symbol is None:
                    # Try to infer from label prefix
                    label_lower = segment.label.lower()
                    if label_lower.startswith("pick"):
                        state_symbol = "a"
                    elif label_lower.startswith("carry") and "empty" in label_lower:
                        state_symbol = "m"
                    elif label_lower.startswith("carry"):
                        state_symbol = "e"
                    elif label_lower.startswith("place"):
                        state_symbol = "i"
                    else:
                        # Default to first state if unknown
                        state_symbol = STATE_SYMBOLS[0]
                        print(f"Warning: Unknown label '{segment.label}', defaulting to '{state_symbol}'")
                
                state_sequence.append(state_symbol)
            else:
                # Timestamp is outside all segments, use last known state or default
                if state_sequence:
                    state_sequence.append(state_sequence[-1])
                else:
                    state_sequence.append(STATE_SYMBOLS[0])
        else:
            # Past all segments, use last known state
            if state_sequence:
                state_sequence.append(state_sequence[-1])
            else:
                state_sequence.append(STATE_SYMBOLS[0])
    
    return state_sequence

