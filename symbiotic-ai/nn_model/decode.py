"""Constrained decoding for legal state sequences."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

from .states import CARRY_START, LEGAL_TRANSITIONS, NUM_STATES, STATE_TO_IDX

DecodeMode = Literal["argmax", "constrained", "count_constrained"]

NEG_INF = -1e30


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """log_softmax along last axis."""
    x = logits - logits.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return x - np.log(exp_x.sum(axis=-1, keepdims=True) + 1e-12)


def decode_argmax(logits: np.ndarray) -> np.ndarray:
    """Per-frame argmax (no transition constraints). logits: (T, C)."""
    return np.argmax(logits, axis=-1).astype(np.int64)


def decode_constrained(logits: np.ndarray, carry_count: Optional[int] = None) -> np.ndarray:
    """
    Viterbi decode with legal transition cycle.

    If carry_count is None, find best legal path (no carry-count constraint).
    If carry_count is K, find best legal path with exactly K CARRY_WITH segments.
    """
    log_probs = _log_softmax(logits.astype(np.float64))
    t_len = log_probs.shape[0]
    if t_len == 0:
        return np.array([], dtype=np.int64)

    if carry_count is None:
        return _viterbi_legal(log_probs)
    return _viterbi_fixed_carry(log_probs, carry_count)


def _viterbi_legal(log_probs: np.ndarray) -> np.ndarray:
    """Standard Viterbi over legal state transitions (no carry-count dimension)."""
    t_len, _ = log_probs.shape
    dp = np.full((t_len, NUM_STATES), NEG_INF, dtype=np.float64)
    back = np.full((t_len, NUM_STATES), -1, dtype=np.int64)

    s0 = STATE_TO_IDX["CARRY_EMPTY"]
    dp[0, s0] = log_probs[0, s0]
    for s in range(NUM_STATES):
        dp[0, s] = max(dp[0, s], log_probs[0, s])

    for t in range(1, t_len):
        for s in range(NUM_STATES):
            emit = log_probs[t, s]
            best = NEG_INF
            best_ps = -1
            for ps in range(NUM_STATES):
                if s not in LEGAL_TRANSITIONS[ps]:
                    continue
                score = dp[t - 1, ps] + emit
                if score > best:
                    best = score
                    best_ps = ps
            dp[t, s] = best
            back[t, s] = best_ps

    end_s = int(np.argmax(dp[t_len - 1]))
    path = np.zeros(t_len, dtype=np.int64)
    path[t_len - 1] = end_s
    for t in range(t_len - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _viterbi_fixed_carry(log_probs: np.ndarray, target_carry: int) -> np.ndarray:
    """
    DP: dp[t][s][k] = best log-score ending at frame t in state s with k carry segments started.
    """
    t_len, _ = log_probs.shape
    max_k = max(target_carry, 0)

    dp = np.full((NUM_STATES, max_k + 1), NEG_INF, dtype=np.float64)
    back_s = np.full((t_len, NUM_STATES, max_k + 1), -1, dtype=np.int64)
    back_k = np.full((t_len, NUM_STATES, max_k + 1), -1, dtype=np.int64)

    s0 = STATE_TO_IDX["CARRY_EMPTY"]
    dp[s0, 0] = log_probs[0, s0]
    for s in range(NUM_STATES):
        if s != s0:
            val = log_probs[0, s]
            if val > dp[s, 0]:
                dp[s, 0] = val

    for t in range(1, t_len):
        new_dp = np.full((NUM_STATES, max_k + 1), NEG_INF, dtype=np.float64)
        for s in range(NUM_STATES):
            for k in range(max_k + 1):
                emit = log_probs[t, s]
                best = NEG_INF
                best_ps, best_pk = -1, -1
                for ps in range(NUM_STATES):
                    if s not in LEGAL_TRANSITIONS[ps]:
                        continue
                    if (ps, s) == CARRY_START:
                        if k == 0:
                            continue
                        pk = k - 1
                    else:
                        pk = k
                    if pk < 0 or pk > max_k:
                        continue
                    score = dp[ps, pk] + emit
                    if score > best:
                        best = score
                        best_ps, best_pk = ps, pk
                if best > NEG_INF:
                    new_dp[s, k] = best
                    back_s[t, s, k] = best_ps
                    back_k[t, s, k] = best_pk
        dp = new_dp

    best_score = NEG_INF
    end_s, end_k = s0, min(target_carry, max_k)
    for s in range(NUM_STATES):
        k = min(target_carry, max_k)
        if dp[s, k] > best_score:
            best_score = dp[s, k]
            end_s, end_k = s, k

    if best_score <= NEG_INF:
        for s in range(NUM_STATES):
            for k in range(max_k + 1):
                if dp[s, k] > best_score:
                    best_score = dp[s, k]
                    end_s, end_k = s, k

    path = np.zeros(t_len, dtype=np.int64)
    path[t_len - 1] = end_s
    ck = end_k
    for t in range(t_len - 1, 0, -1):
        s = int(path[t])
        ps = back_s[t, s, ck]
        pk = back_k[t, s, ck]
        if ps < 0:
            path[t - 1] = s
        else:
            path[t - 1] = ps
            ck = pk
    return path


def count_carry_segments(path: np.ndarray) -> int:
    """Count CARRY_WITH segments (transitions PICK -> CARRY_WITH)."""
    carry_idx = STATE_TO_IDX["CARRY_WITH"]
    pick_idx = STATE_TO_IDX["PICK"]
    count = 0
    for t in range(1, len(path)):
        if path[t - 1] == pick_idx and path[t] == carry_idx:
            count += 1
    if len(path) > 0 and path[0] == carry_idx:
        count += 1
    return count


def run_decode(
    logits: np.ndarray,
    decode_mode: DecodeMode,
    carry_count: Optional[int],
) -> Tuple[np.ndarray, DecodeMode]:
    """Decode logits, falling back from count_constrained to constrained if K is missing."""
    if decode_mode == "count_constrained":
        if carry_count is None:
            return decode(logits, mode="constrained"), "constrained"
        return decode(logits, mode="count_constrained", carry_count=carry_count), "count_constrained"
    return decode(logits, mode=decode_mode), decode_mode


def decode(
    logits: np.ndarray,
    mode: DecodeMode = "constrained",
    carry_count: Optional[int] = None,
) -> np.ndarray:
    if mode == "argmax":
        return decode_argmax(logits)
    if mode == "constrained":
        return decode_constrained(logits, carry_count=None)
    if mode == "count_constrained":
        if carry_count is None:
            raise ValueError("count_constrained mode requires carry_count")
        return decode_constrained(logits, carry_count=carry_count)
    raise ValueError(f"Unknown decode mode: {mode!r}")
