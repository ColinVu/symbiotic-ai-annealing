"""Forced alignment over frame-level posteriors using a left-to-right HMM.

Given per-frame class posteriors from a cross-entropy-trained network and a
known label sequence, build a linear left-to-right chain whose j-th state emits
label l_j, then Viterbi-decode. Every frame receives a label, so unlike CTC
there is no blank policy and no post-hoc gap filling.


A NOTE ON TRANSITION PROBABILITIES
----------------------------------
They contribute almost nothing here, and it is worth understanding why before
reaching for them.

1. A *homogeneous* transition model cancels exactly. Every valid path through
   the chain makes exactly S-1 advances and T-S stays -- those counts are fixed
   by the topology, not by the path -- so a transition cost that is identical
   everywhere adds the same constant to every candidate and cannot change the
   argmax. An earlier version of this module had an `advance_penalty` argument
   that was provably a no-op for this reason. It has been removed.

2. A *state-dependent* model does not cancel: the path-dependent term becomes
   sum_j n_j * log p_{c_j}, which varies with how frames are allocated. But
   measured on real-scale data the spread in log p across states is around
   0.08 log units per frame, while the emission gap between the best and
   second-best class is around 1.7 -- roughly 20x larger. The duration prior
   is too weak to flip decisions, and amplifying it via transition_scale makes
   alignment monotonically worse, because it starts overriding real evidence.

3. Enabling the state-dependent model at its correct scale of 1.0 measurably
   *reduced* downstream task accuracy on real data, relative to running with
   no transition term. The reason is visible in a worked example: the duration
   prior pulls each segment toward that label's *average* length, which is the
   wrong instinct whenever a particular video deviates from average -- and the
   deviations are exactly what we are trying to measure.

This is a known property of hybrid HMMs rather than a defect here: once
emissions come from a discriminative network they dominate, and geometric
duration modelling is too crude to add much.

`transition_scale` therefore defaults to 0. The machinery is retained so the
result stays reproducible, not because it is recommended.

Duration is therefore controlled by `min_duration`, a hard structural
constraint that evidence cannot override. The transition machinery is retained
because it is the textbook-correct formulation and because being able to
demonstrate that it does not help is worth more than omitting it.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

NEG_INF = -1e30


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max(axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


# --------------------------------------------------------------------------- #
# topology
# --------------------------------------------------------------------------- #


def build_chain(
    targets: Sequence[int], min_duration: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand a label sequence into HMM chain states.

    Returns (chain, allow_self): chain[j] is the emitting class of chain state
    j, and allow_self[j] says whether that state has a self-loop. Each label
    becomes `min_duration` chained sub-states, only the last of which may
    self-loop, so the path must spend at least min_duration frames per label.

    Duration is then a shifted geometric: min_duration - 1 forced frames plus
    Geometric(1 - p) further frames.
    """
    if min_duration < 1:
        raise ValueError("min_duration must be >= 1")
    chain: List[int] = []
    allow_self: List[bool] = []
    for label in targets:
        for k in range(min_duration):
            chain.append(int(label))
            allow_self.append(k == min_duration - 1)
    return np.asarray(chain, dtype=np.int64), np.asarray(allow_self, dtype=bool)


# --------------------------------------------------------------------------- #
# transition estimation
# --------------------------------------------------------------------------- #


def segments_from_frames(frame_labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """(start, end_exclusive, class) runs from dense frame labels."""
    frame_labels = np.asarray(frame_labels)
    if frame_labels.size == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    start = 0
    for t in range(1, len(frame_labels)):
        if frame_labels[t] != frame_labels[start]:
            out.append((start, t, int(frame_labels[start])))
            start = t
    out.append((start, len(frame_labels), int(frame_labels[start])))
    return out


def duration_stats(
    label_arrays: Iterable[np.ndarray], num_classes: int
) -> Dict[str, np.ndarray]:
    """Per-class segment count and total frames, from dense frame labels."""
    counts = np.zeros(num_classes, dtype=np.float64)   # k: number of segments
    totals = np.zeros(num_classes, dtype=np.float64)   # N: total frames
    for arr in label_arrays:
        arr = np.asarray(arr)
        arr = arr[arr >= 0]
        if arr.size == 0:
            continue
        for start, end, cls in segments_from_frames(arr):
            if 0 <= cls < num_classes:
                counts[cls] += 1
                totals[cls] += end - start
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dur = np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)
    return {"n_segments": counts, "n_frames": totals, "mean_duration": mean_dur}


def estimate_self_loop_logprobs(
    label_arrays: Iterable[np.ndarray],
    num_classes: int,
    default_duration: float = 10.0,
) -> np.ndarray:
    """Maximum-likelihood self-loop log-probabilities from observed durations.

    A segment of length n has probability p^(n-1) * (1-p). Over k segments
    totalling N frames the log-likelihood is (N-k) log p + k log(1-p), whose
    stationary point is

        p = 1 - k/N = 1 - 1/mean_duration

    so the estimator is one minus the reciprocal of the mean segment length.
    Classes never observed fall back to `default_duration`.
    """
    stats = duration_stats(label_arrays, num_classes)
    mean_dur = stats["mean_duration"].copy()
    mean_dur[mean_dur < 1.0000001] = default_duration
    p = 1.0 - 1.0 / mean_dur
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p)


# --------------------------------------------------------------------------- #
# emissions
# --------------------------------------------------------------------------- #


def emission_scores(
    logits: np.ndarray,
    priors: Optional[np.ndarray] = None,
    prior_scale: float = 1.0,
) -> np.ndarray:
    """(T, C) logits -> (T, C) scaled log likelihoods.

    Bayes gives p(x|s) = p(s|x) p(x) / p(s). The p(x) factor is identical
    across states at a given frame and drops out of the argmax, leaving
    posterior divided by prior. Without the division, frequent states gain an
    unearned advantage at every frame and eat into their neighbours at
    boundaries.
    """
    logp = log_softmax(np.asarray(logits, dtype=np.float64))
    if priors is None or prior_scale == 0.0:
        return logp
    pri = np.asarray(priors, dtype=np.float64)
    if pri.shape[0] != logp.shape[1]:
        raise ValueError(f"priors has {pri.shape[0]} entries, logits have {logp.shape[1]} classes")
    pri = np.clip(pri, 1e-8, None)
    pri = pri / pri.sum()
    return logp - prior_scale * np.log(pri)[None, :]


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #


def transition_costs(
    chain: np.ndarray,
    allow_self: np.ndarray,
    self_loop_logprobs: Optional[np.ndarray],
    transition_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-chain-state (stay_cost, advance_cost) in log space.

    advance_cost[j] is the cost of arriving at j from j-1, i.e. the cost of
    leaving j-1. Only a self-looping state pays an exit cost of log(1-p);
    advancing between the forced sub-states of one label is deterministic and
    costs nothing, which keeps the duration model a correctly normalised
    shifted geometric rather than charging the exit min_duration times.
    """
    S = len(chain)
    if self_loop_logprobs is None or transition_scale == 0.0:
        return np.zeros(S), np.zeros(S)

    slp = np.asarray(self_loop_logprobs, dtype=np.float64)
    p = np.clip(np.exp(slp), 1e-6, 1.0 - 1e-6)
    log_p = transition_scale * np.log(p)
    log_1mp = transition_scale * np.log1p(-p)

    stay_cost = log_p[chain]
    advance_cost = np.zeros(S)
    for j in range(1, S):
        advance_cost[j] = log_1mp[chain[j - 1]] if allow_self[j - 1] else 0.0
    return stay_cost, advance_cost


def viterbi_align(
    logits: np.ndarray,
    targets: Sequence[int],
    priors: Optional[np.ndarray] = None,
    prior_scale: float = 1.0,
    min_duration: int = 1,
    self_loop_logprobs: Optional[np.ndarray] = None,
    transition_scale: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Align `targets` to frames through a left-to-right HMM.

    Args:
        logits: (T, C) raw network outputs.
        targets: label sequence (class indices) in order.
        priors: (C,) class priors from training; None disables the division.
        prior_scale: exponent on the prior. 0 = plain posteriors.
        min_duration: minimum frames per label. This, not the transition
            model, is what actually controls segment length.
        self_loop_logprobs: (C,) log p(stay) per class, from
            estimate_self_loop_logprobs. None means no transition term.
        transition_scale: multiplier on the transition log-probabilities.
            Defaults to 0 (transitions disabled) because enabling them
            measurably reduced downstream task accuracy on real data. 1.0 is
            the correctly normalised model; see module docstring.

    Returns:
        (frame_labels, score): frame_labels is (T,) of class indices, one per
        frame; score is the path log-score.
    """
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"expected (T, C) logits, got {logits.shape}")
    targets = list(targets)
    if not targets:
        raise ValueError("targets must be non-empty")

    emis = emission_scores(logits, priors, prior_scale)
    T, C = emis.shape
    if max(targets) >= C or min(targets) < 0:
        raise ValueError(f"target index out of range for C={C}")

    chain, allow_self = build_chain(targets, min_duration)
    S = len(chain)
    if T < S:
        raise ValueError(
            f"{len(targets)} labels at min_duration={min_duration} need {S} frames, got T={T}"
        )

    stay_cost, advance_cost = transition_costs(
        chain, allow_self, self_loop_logprobs, transition_scale
    )

    delta = np.full(S, NEG_INF)
    delta[0] = emis[0, chain[0]]
    back = np.zeros((T, S), dtype=np.uint8)  # 1 = arrived by advancing
    chain_emis = emis[:, chain]

    for t in range(1, T):
        stay = np.where(allow_self, delta + stay_cost, NEG_INF)
        adv = np.full(S, NEG_INF)
        adv[1:] = delta[:-1] + advance_cost[1:]
        took_adv = adv > stay
        delta = np.where(took_adv, adv, stay) + chain_emis[t]
        back[t] = took_adv

    score = float(delta[S - 1])
    if score <= NEG_INF / 2:
        raise ValueError("no valid alignment path; check min_duration against T")

    frame_states = np.zeros(T, dtype=np.int64)
    j = S - 1
    for t in range(T - 1, -1, -1):
        frame_states[t] = j
        if t > 0 and back[t, j]:
            j -= 1
    return chain[frame_states], score


def boundary_errors(pred: np.ndarray, ref: np.ndarray) -> List[int]:
    """Signed frame offsets between matched segment starts of pred and ref.

    Returns [] when the segment counts differ, since the pairing is then
    undefined.
    """
    ps = segments_from_frames(pred)
    rs = segments_from_frames(ref)
    if len(ps) != len(rs):
        return []
    return [p[0] - r[0] for p, r in zip(ps, rs)][1:]
