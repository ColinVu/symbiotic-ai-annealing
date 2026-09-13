"""State token and label constants for pick/carry/place/empty classification."""

from __future__ import annotations

TOKEN_TO_STATE = {
    "a": "PICK",
    "e": "CARRY_WITH",
    "i": "PLACE",
    "m": "CARRY_EMPTY",
    "sil": "SIL",
}

STATE_TO_TOKEN = {v: k for k, v in TOKEN_TO_STATE.items() if k != "sil"}

STATES = ("CARRY_EMPTY", "PICK", "CARRY_WITH", "PLACE")
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
IDX_TO_STATE = {i: s for s, i in STATE_TO_IDX.items()}

NUM_STATES = len(STATES)

# Legal forward transitions (including self-loop): prev -> {next, ...}
LEGAL_TRANSITIONS: dict[int, set[int]] = {
    STATE_TO_IDX["CARRY_EMPTY"]: {STATE_TO_IDX["CARRY_EMPTY"], STATE_TO_IDX["PICK"]},
    STATE_TO_IDX["PICK"]: {STATE_TO_IDX["PICK"], STATE_TO_IDX["CARRY_WITH"]},
    STATE_TO_IDX["CARRY_WITH"]: {STATE_TO_IDX["CARRY_WITH"], STATE_TO_IDX["PLACE"]},
    STATE_TO_IDX["PLACE"]: {STATE_TO_IDX["PLACE"], STATE_TO_IDX["CARRY_EMPTY"]},
}

# Transition that starts a new carry segment (increments carry count).
CARRY_START = (STATE_TO_IDX["PICK"], STATE_TO_IDX["CARRY_WITH"])
