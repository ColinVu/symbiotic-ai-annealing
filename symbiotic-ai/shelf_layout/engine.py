"""Capacity-preserving shelf recommendations from observations revealed so far.

The decision engine never reads ground truth. Its inferred SKU assignments are
weak supervision; all reported error changes are in-simulation proxy estimates.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np


class SnapshotError(ValueError):
    """A model or simulation snapshot is incomplete or inconsistent."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)).encode()).hexdigest()


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise SnapshotError("Embedding or centroid has a zero or non-finite norm")
    return matrix / norms


@dataclass(frozen=True)
class ModelData:
    labels: Tuple[str, ...]
    centroids: np.ndarray
    transform: Callable[[np.ndarray], np.ndarray]
    fingerprint: str


def load_model(model_dir: Union[str, Path]) -> ModelData:
    """Load saved centroid files and the inference-time optional transforms.

    CLIP itself is unnecessary because snapshots contain frame embeddings.
    """
    root = Path(model_dir)
    metadata_path, array_path = root / "model_metadata.json", root / "centroids.npy"
    missing = [str(p) for p in (metadata_path, array_path) if not p.is_file()]
    if missing:
        raise SnapshotError("Missing model input: " + ", ".join(missing))
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    original_labels = [str(x) for x in meta.get("centroid_labels", [])]
    if not original_labels or len(set(original_labels)) != len(original_labels):
        raise SnapshotError("Model metadata must list unique centroid_labels")
    raw = np.load(array_path, allow_pickle=False)
    if raw.ndim != 2 or raw.shape[0] != len(original_labels) or not np.isfinite(raw).all():
        raise SnapshotError("centroids.npy shape or values do not match model metadata")
    order = np.argsort(np.asarray(original_labels), kind="stable")
    labels = tuple(original_labels[int(i)] for i in order)
    centroids = _unit(np.asarray(raw[order], dtype=np.float64))

    transforms: List[Callable[[np.ndarray], np.ndarray]] = []
    hn_path = root / "hand_neutralizer.json"
    if hn_path.is_file():
        state = json.loads(hn_path.read_text(encoding="utf-8"))
        if state.get("enabled"):
            mean = np.asarray(state.get("mean"), dtype=np.float64)
            components = np.asarray(state.get("components"), dtype=np.float64)
            if mean.shape != (raw.shape[1],) or components.ndim != 2 or components.shape[1] != raw.shape[1]:
                raise SnapshotError("Invalid hand_neutralizer.json dimensions")
            def neutralize(x: np.ndarray, mean: np.ndarray = mean, components: np.ndarray = components) -> np.ndarray:
                centered = x - mean
                return centered - (centered @ components.T) @ components
            transforms.append(neutralize)
    adapter_path = root / "clip_adapter.pt"
    if adapter_path.is_file():
        try:
            import torch
            from symbiote_weak_generalized.training.clip_adapter import CLIPAdapter
        except ImportError as exc:
            raise SnapshotError("clip_adapter.pt requires torch and project ML dependencies") from exc
        adapter = CLIPAdapter(int(meta.get("embedding_dim", raw.shape[1])))
        adapter.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True))
        adapter.eval()
        def adapt(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return adapter(torch.from_numpy(x).float()).numpy().astype(np.float64)
        transforms.append(adapt)

    def transform(x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float64)
        for fn in transforms:
            out = fn(out)
        return out

    files = [metadata_path, array_path] + [p for p in (hn_path, adapter_path) if p.is_file()]
    hash_value = hashlib.sha256()
    for path in files:
        hash_value.update(path.name.encode())
        hash_value.update(path.read_bytes())
    return ModelData(labels, centroids, transform, hash_value.hexdigest())


def validate_snapshot(snapshot: Mapping[str, Any], model: ModelData) -> Dict[str, Any]:
    """Validate and normalize the explicit simulator-to-decision API payload."""
    if not isinstance(snapshot, Mapping):
        raise SnapshotError("Snapshot must be a JSON object")
    if (model.centroids.ndim != 2 or model.centroids.shape[0] != len(model.labels)
            or len(set(model.labels)) != len(model.labels) or not np.isfinite(model.centroids).all()):
        raise SnapshotError("ModelData has invalid labels or centroid matrix")
    forbidden = {"ground_truth", "true_label", "truth", "expected_label"}
    if forbidden.intersection(snapshot):
        raise SnapshotError("Decision snapshot must not contain ground-truth labels")
    step = snapshot.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise SnapshotError("step must be a nonnegative integer")
    capacities = snapshot.get("capacities")
    if not isinstance(capacities, dict) or not capacities or any(
        not isinstance(s, str) or not isinstance(c, int) or isinstance(c, bool) or c < 3
        for s, c in capacities.items()
    ):
        raise SnapshotError("capacities must map shelf names to integers >= 3")
    raw_items = snapshot.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != sum(capacities.values()):
        raise SnapshotError("items must fill every declared shelf slot")
    locations: Dict[str, Tuple[str, str]] = {}
    occupied = set()
    counts: Counter[str] = Counter()
    for item in raw_items:
        if not isinstance(item, dict):
            raise SnapshotError("Each item must be an object")
        sku, shelf, slot = item.get("sku"), item.get("shelf"), item.get("slot")
        if not isinstance(sku, str) or not sku or shelf not in capacities or not isinstance(slot, (str, int)):
            raise SnapshotError("Every item needs sku, valid shelf, and slot")
        slot = str(slot)
        if not slot or sku in locations or (shelf, slot) in occupied:
            raise SnapshotError("Duplicate SKU or occupied shelf slot")
        locations[sku] = (shelf, slot)
        occupied.add((shelf, slot))
        counts[shelf] += 1
    if dict(counts) != capacities:
        raise SnapshotError("Shelf occupancy does not match capacities")
    if set(locations) != set(model.labels):
        missing = sorted(set(locations) - set(model.labels))
        extra = sorted(set(model.labels) - set(locations))
        raise SnapshotError(f"Model/inventory SKU mismatch: no centroid={missing}, no item={extra}")

    raw_observations = snapshot.get("observations", [])
    if not isinstance(raw_observations, list):
        raise SnapshotError("observations must be a list")
    observations = []
    ids = set()
    for row in raw_observations:
        if not isinstance(row, dict) or forbidden.intersection(row):
            raise SnapshotError("Observation must be an object without ground-truth labels")
        oid, sku, revealed = row.get("id"), row.get("inferred_sku"), row.get("revealed_at")
        observed_shelf = row.get("observed_shelf")
        if not isinstance(oid, str) or not oid or oid in ids:
            raise SnapshotError("Observation IDs must be unique nonempty strings")
        if sku not in locations:
            raise SnapshotError(f"Unknown inferred_sku in observation {oid}")
        if observed_shelf not in capacities:
            raise SnapshotError(f"Observation {oid} needs an observed_shelf from its pickup event")
        if observed_shelf != locations[sku][0]:
            raise SnapshotError(
                f"Observation {oid} assigns {sku} to a different shelf than its observed pickup; "
                "resolve the weak assignment before using it for a move decision"
            )
        if not isinstance(revealed, int) or isinstance(revealed, bool) or revealed < 0 or revealed > step:
            raise SnapshotError(f"Observation {oid} is not revealed at step {step}")
        frames = np.asarray(row.get("frames", []), dtype=np.float64)
        if frames.size == 0:
            frames = np.empty((0, model.centroids.shape[1]), dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != model.centroids.shape[1] or not np.isfinite(frames).all():
            raise SnapshotError(f"Observation {oid} has invalid frame embeddings")
        if frames.size:
            frames = model.transform(frames)
            if frames.shape != (len(row["frames"]), model.centroids.shape[1]) or not np.isfinite(frames).all():
                raise SnapshotError(f"Observation {oid} failed model transforms")
            frames = _unit(frames)
        conf = row.get("assignment_confidence")
        if conf is not None and (not isinstance(conf, (float, int)) or not 0 <= conf <= 1):
            raise SnapshotError(f"Observation {oid} has invalid assignment confidence")
        observations.append({"id": oid, "sku": sku, "observed_shelf": observed_shelf,
                             "revealed_at": revealed, "frames": frames, "confidence": conf})
        ids.add(oid)
    return {"step": step, "capacities": dict(capacities), "locations": locations, "observations": observations,
            "previous_decision": snapshot.get("previous_decision"), "snapshot_hash": _digest(snapshot)}


def _margin(distances: np.ndarray) -> np.ndarray:
    """Vector form of evaluator._harmonic_three_centroid_margin."""
    if distances.shape[1] < 2:
        return np.zeros(len(distances))
    first = distances[:, 0]
    m12 = np.maximum(0.0, distances[:, 1] - first)
    if distances.shape[1] < 3:
        return m12
    m13 = np.maximum(0.0, distances[:, 2] - first)
    both = (m12 > 0) & (m13 > 0)
    out = np.where(m12 <= 0, m13, np.where(m13 <= 0, m12, 0.0))
    out[both] = 2 * m12[both] * m13[both] / (m12[both] + m13[both])
    return out


class DecisionEngine:
    """Immutable-snapshot analysis and deterministic capacity-preserving search."""

    def __init__(self, snapshot: Mapping[str, Any], model: ModelData):
        self.model = model
        self.data = validate_snapshot(snapshot, model)
        self.labels = model.labels
        self.label_index = {sku: i for i, sku in enumerate(self.labels)}
        self.baseline = {sku: shelf for sku, (shelf, _) in self.data["locations"].items()}
        self.observations = self.data["observations"]
        self.distances: Dict[str, np.ndarray] = {}
        for row in self.observations:
            self.distances[row["id"]] = 1.0 - row["frames"] @ model.centroids.T
        self._shelf_cache: Dict[Tuple[str, ...], Tuple[int, Dict[str, str]]] = {}
        self.pair_risk = self._pair_risk()

    def _pair_risk(self) -> np.ndarray:
        """Directed observed overlap, with each segment contributing total weight one."""
        risk = np.zeros((len(self.labels), len(self.labels)), dtype=np.float64)
        for row in self.observations:
            d = self.distances[row["id"]]
            if not len(d):
                continue
            own = self.label_index[row["sku"]]
            risk[own] += np.mean(d <= d[:, own:own + 1], axis=0)
        np.fill_diagonal(risk, 0.0)
        self.directed_pair_risk = risk
        return risk + risk.T

    def _risk_totals(self, layout: Mapping[str, str]) -> Dict[str, np.ndarray]:
        """Risk from each SKU to the current occupants of each shelf."""
        return {shelf: self.pair_risk[:, [self.label_index[x] for x in self.labels
                                          if layout[x] == shelf]].sum(axis=1)
                for shelf in self.data["capacities"]}

    def _swap_proxy_delta(self, layout: Mapping[str, str], a: str, b: str,
                          totals: Optional[Mapping[str, np.ndarray]] = None) -> float:
        """Fast pairwise risk change used only to prioritize exact replay."""
        shelf_a, shelf_b = layout[a], layout[b]
        ia, ib = self.label_index[a], self.label_index[b]
        risk = self.pair_risk
        totals = totals if totals is not None else self._risk_totals(layout)
        return float(totals[shelf_a][ib] - risk[ib, ia] + totals[shelf_b][ia] - risk[ia, ib]
                     - totals[shelf_a][ia] - totals[shelf_b][ib])

    def _confusion_evidence(self, layout: Mapping[str, str], predictions: Mapping[str, str],
                            min_pair_segments: int = 5, min_pair_confusions: int = 2,
                            min_pair_rate: float = 0.20) -> List[Dict[str, Any]]:
        """Observed directed errors, grouped by unordered pair for search."""
        counts: Counter[str] = Counter()
        confusions: Counter[Tuple[str, str]] = Counter()
        confidence: Dict[str, List[float]] = defaultdict(list)
        for row in self.observations:
            if not len(self.distances[row["id"]]):
                continue
            sku = row["sku"]
            counts[sku] += 1
            if row["confidence"] is not None:
                confidence[sku].append(float(row["confidence"]))
            pred = predictions[row["id"]]
            if pred and pred != sku:
                confusions[(sku, pred)] += 1
        pairs = []
        for a, b in sorted({tuple(sorted((source, target))) for source, target in confusions}):
            total = confusions[a, b] + confusions[b, a]
            segments = counts[a] + counts[b]
            rate = total / segments if segments else 0.0
            reasons = []
            if layout[a] != layout[b]:
                reasons.append("different_shelves")
            if segments < min_pair_segments:
                reasons.append("insufficient_segments")
            if total < min_pair_confusions:
                reasons.append("insufficient_confusions")
            if rate < min_pair_rate:
                reasons.append("low_rate")
            known_confidence = confidence[a] + confidence[b]
            pairs.append({"a": a, "b": b, "a_to_b": confusions[a, b],
                          "b_to_a": confusions[b, a], "a_segments": counts[a],
                          "b_segments": counts[b], "pair_count": total,
                          "pair_rate": rate, "same_shelf": layout[a] == layout[b],
                          "mean_assignment_confidence": (sum(known_confidence) / len(known_confidence)
                                                         if known_confidence else None),
                          "confidence_segments": len(known_confidence),
                          "eligible": not reasons, "exclusion_reasons": reasons})
        return sorted(pairs, key=lambda row: (-row["pair_count"], -row["pair_rate"],
                                              row["a"], row["b"]))

    def _fallback_pairs(self, layout: Mapping[str, str], budget: int) -> List[Tuple[str, str]]:
        """Select a bounded set of risky same-shelf pairs without cross-shelf enumeration."""
        if budget <= 0:
            return []
        candidates = []
        for shelf in self.data["capacities"]:
            members = [sku for sku in self.labels if layout[sku] == shelf]
            for a, b in combinations(members, 2):
                candidates.append((-float(self.pair_risk[self.label_index[a], self.label_index[b]]), a, b))
        return [(a, b) for risk, a, b in sorted(candidates)[:budget] if risk < 0]

    def _targeted_swaps(self, layout: Mapping[str, str], pairs: List[Tuple[str, str]],
                        partners_per_shelf: int, max_moved: int,
                        totals: Mapping[str, np.ndarray]) -> List[Tuple[float, str, str, Tuple[str, str]]]:
        """Bounded proxy-ranked cross-shelf swaps around nominated items."""
        proposals: Dict[Tuple[str, str], Tuple[float, str, str, Tuple[str, str]]] = {}
        moved_now = sum(layout[x] != self.baseline[x] for x in self.labels)
        for pair in pairs:
            for source in pair:
                for shelf in sorted(self.data["capacities"]):
                    if shelf == layout[source]:
                        continue
                    ranked = sorted((self._swap_proxy_delta(layout, source, partner, totals), partner)
                                    for partner in self.labels if layout[partner] == shelf)
                    kept = 0
                    for delta, partner in ranked:
                        moved_after = moved_now
                        moved_after += (layout[partner] != self.baseline[source]) - (layout[source] != self.baseline[source])
                        moved_after += (layout[source] != self.baseline[partner]) - (layout[partner] != self.baseline[partner])
                        if moved_after > max_moved:
                            continue
                        a, b = sorted((source, partner))
                        proposal = (delta, a, b, pair)
                        if (a, b) not in proposals or proposal < proposals[a, b]:
                            proposals[a, b] = proposal
                        kept += 1
                        if kept >= partners_per_shelf:
                            break
        return sorted(proposals.values())

    def _members(self, layout: Mapping[str, str]) -> Dict[str, Tuple[str, ...]]:
        return {s: tuple(sku for sku in self.labels if layout[sku] == s)
                for s in sorted(self.data["capacities"])}

    def _predict(self, oid: str, members: Tuple[str, ...]) -> str:
        distances = self.distances[oid]
        if not len(distances):
            return ""
        indices = np.array([self.label_index[sku] for sku in members], dtype=int)
        candidates = distances[:, indices]
        ranked = np.argsort(candidates, axis=1, kind="stable")[:, :3]
        d = np.take_along_axis(candidates, ranked, axis=1)
        weights = _margin(d)
        totals: Dict[str, float] = defaultdict(float)
        for i, weight in zip(ranked[:, 0], weights):
            totals[members[int(i)]] += float(weight)
        return min(totals, key=lambda sku: (-totals[sku], sku))

    def _shelf_score(self, members: Tuple[str, ...]) -> Tuple[int, Dict[str, str]]:
        if members in self._shelf_cache:
            return self._shelf_cache[members]
        member_set = set(members)
        predictions = {}
        errors = 0
        for row in self.observations:
            if row["sku"] not in member_set:
                continue
            pred = self._predict(row["id"], members)
            predictions[row["id"]] = pred
            errors += pred != row["sku"]
        result = (errors, predictions)
        self._shelf_cache[members] = result
        return result

    def score(self, layout: Mapping[str, str]) -> Dict[str, Any]:
        """Counterfactual replay: assumes each inferred SKU is picked from its proposed shelf."""
        if set(layout) != set(self.labels):
            raise SnapshotError("Layout must assign every SKU exactly once")
        members = self._members(layout)
        if any(len(members[s]) != cap for s, cap in self.data["capacities"].items()):
            raise SnapshotError("Layout violates shelf capacities")
        predictions: Dict[str, str] = {}
        errors = 0
        for group in members.values():
            cost, pred = self._shelf_score(group)
            errors += cost
            predictions.update(pred)
        return {"errors": errors, "count": len(self.observations), "accuracy":
                (len(self.observations) - errors) / len(self.observations) if self.observations else None,
                "predictions": predictions}

    def score_current(self) -> Dict[str, Any]:
        """Actual current replay: restrict each pick to its observed pickup shelf."""
        members = self._members(self.baseline)
        predictions = {row["id"]: self._predict(row["id"], members[row["observed_shelf"]])
                       for row in self.observations}
        errors = sum(predictions[row["id"]] != row["sku"] for row in self.observations)
        return {"errors": errors, "count": len(self.observations),
                "accuracy": (len(self.observations) - errors) / len(self.observations) if self.observations else None,
                "predictions": predictions}

    def _objective(self, layout: Mapping[str, str], move_cost: float) -> Tuple[float, int, int]:
        moved = sum(layout[sku] != shelf for sku, shelf in self.baseline.items())
        errors = (self.score_current() if moved == 0 else self.score(layout))["errors"]
        return (errors + move_cost * moved, moved, errors)

    def _cycle_pool(self, predictions: Mapping[str, str], layout: Mapping[str, str],
                    limit: int = 12) -> List[str]:
        risk = Counter()
        for row in self.observations:
            pred = predictions[row["id"]]
            if pred != row["sku"]:
                risk[row["sku"]] += 2
                if pred:
                    risk[pred] += 1
        selected = [sku for sku, _ in sorted(risk.items(), key=lambda x: (-x[1], x[0]))[:max(1, limit // 2)]]
        if selected:
            similarities = self.model.centroids @ self.model.centroids.T
            for sku in list(selected):
                i = self.label_index[sku]
                for j in np.argsort(-similarities[i], kind="stable"):
                    other = self.labels[int(j)]
                    if layout[other] != layout[sku] and other not in selected:
                        selected.append(other)
                        break
                if len(selected) >= limit:
                    break
        return sorted(selected[:limit])

    def recommend(self, *, move_cost: float = 0.25, max_moved: int = 12,
                  cycle_pool_size: int = 12, swap_replay_limit: int = 200,
                  search_mode: str = "confusion", min_pair_segments: int = 5,
                  min_pair_confusions: int = 2, min_pair_rate: float = 0.20,
                  max_confusion_pairs: int = 10, partners_per_shelf: int = 5,
                  fallback_swap_budget: int = 32) -> Dict[str, Any]:
        if (not np.isfinite(move_cost) or move_cost < 0 or max_moved < 0 or cycle_pool_size < 0
                or swap_replay_limit < 1 or min_pair_segments < 1 or min_pair_confusions < 1
                or not np.isfinite(min_pair_rate) or not 0 <= min_pair_rate <= 1
                or max_confusion_pairs < 1 or partners_per_shelf < 1 or fallback_swap_budget < 0
                or search_mode not in {"confusion", "all_pairs"}):
            raise ValueError("Invalid search settings")
        current = dict(self.baseline)
        baseline_score = self.score_current()
        best_seen = {tuple(current[sku] for sku in self.labels): (self._objective(current, move_cost), dict(current))}
        seen = set(best_seen)
        search_stats = {"passes": [], "total_generated_swaps": 0, "total_exact_swap_replays": 0,
                        "fallback_passes": 0, "no_move_reason": None}
        while True:
            old_obj = self._objective(current, move_cost)
            choice = None
            predictions = (baseline_score if current == self.baseline else self.score(current))["predictions"]
            evidence = self._confusion_evidence(current, predictions, min_pair_segments,
                                                min_pair_confusions, min_pair_rate)
            triggers = [row for row in evidence if row["eligible"]][:max_confusion_pairs]
            stats = {"trigger_pairs": triggers, "observed_pairs": evidence,
                     "generated_swaps": 0, "exact_swap_replays": 0,
                     "fallback_used": False, "fallback_generated_swaps": 0,
                     "replayed_swaps": []}
            totals = self._risk_totals(current)
            if search_mode == "all_pairs":
                swap_queue = []
                for a, b in combinations(self.labels, 2):
                    if current[a] == current[b]:
                        continue
                    candidate = dict(current)
                    candidate[a], candidate[b] = candidate[b], candidate[a]
                    key = tuple(candidate[sku] for sku in self.labels)
                    if key in seen or sum(candidate[x] != self.baseline[x] for x in self.labels) > max_moved:
                        continue
                    swap_queue.append((self._swap_proxy_delta(current, a, b, totals), key, a, b, candidate))
                stats["generated_swaps"] = len(swap_queue)
                ranked_candidates = [
                    (key, candidate, {"source": "all_pairs", "a": a, "b": b,
                                      "proxy_delta": float(delta), "proxy_rank": rank})
                    for rank, (delta, key, a, b, candidate) in enumerate(
                        sorted(swap_queue, key=lambda x: (x[0], x[1]))[:swap_replay_limit], 1)
                ]
            else:
                proposals = self._targeted_swaps(current, [(r["a"], r["b"]) for r in triggers],
                                                  partners_per_shelf, max_moved, totals)
                stats["generated_swaps"] = len(proposals)
                ranked_candidates = []
                for rank, (delta, a, b, pair) in enumerate(proposals, 1):
                    candidate = dict(current)
                    candidate[a], candidate[b] = candidate[b], candidate[a]
                    key = tuple(candidate[sku] for sku in self.labels)
                    if key not in seen:
                        ranked_candidates.append((key, candidate,
                                                  {"source": "confusion_pair", "a": a, "b": b,
                                                   "trigger_pair": list(pair), "proxy_delta": float(delta),
                                                   "proxy_rank": rank}))
                    if len(ranked_candidates) >= swap_replay_limit:
                        break
            for key, candidate, provenance in ranked_candidates:
                obj = self._objective(candidate, move_cost)
                stats["exact_swap_replays"] += 1
                stats["replayed_swaps"].append({**provenance, "objective": obj[0],
                                                 "proxy_errors": obj[2]})
                if obj < old_obj and (choice is None or (obj, key) < (choice[0], choice[1])):
                    choice = (obj, key, candidate)
                best_seen[key] = (obj, candidate)
            if search_mode == "confusion" and choice is None and fallback_swap_budget:
                stats["fallback_used"] = True
                search_stats["fallback_passes"] += 1
                fallback_pairs = self._fallback_pairs(current, fallback_swap_budget)
                fallback = self._targeted_swaps(current, fallback_pairs, partners_per_shelf,
                                                max_moved, totals)
                stats["fallback_generated_swaps"] = len(fallback)
                remaining = min(fallback_swap_budget, swap_replay_limit - stats["exact_swap_replays"])
                for rank, (delta, a, b, pair) in enumerate(fallback, 1):
                    if remaining <= 0:
                        break
                    candidate = dict(current)
                    candidate[a], candidate[b] = candidate[b], candidate[a]
                    key = tuple(candidate[sku] for sku in self.labels)
                    if key in seen or key in best_seen:
                        continue
                    obj = self._objective(candidate, move_cost)
                    stats["exact_swap_replays"] += 1
                    stats["replayed_swaps"].append({"source": "fallback", "a": a, "b": b,
                                                    "trigger_pair": list(pair), "proxy_delta": float(delta),
                                                    "proxy_rank": rank, "objective": obj[0],
                                                    "proxy_errors": obj[2]})
                    remaining -= 1
                    if obj < old_obj and (choice is None or (obj, key) < (choice[0], choice[1])):
                        choice = (obj, key, candidate)
                    best_seen[key] = (obj, candidate)
            if choice is None and cycle_pool_size >= 3:
                pool = self._cycle_pool(predictions, current, cycle_pool_size)
                for triple in combinations(pool, 3):
                    if len({current[x] for x in triple}) != 3:
                        continue
                    for rotated in permutations([current[x] for x in triple]):
                        if any(rotated[i] == current[x] for i, x in enumerate(triple)):
                            continue
                        candidate = dict(current)
                        candidate.update(zip(triple, rotated))
                        key = tuple(candidate[sku] for sku in self.labels)
                        if key in seen or sum(candidate[x] != self.baseline[x] for x in self.labels) > max_moved:
                            continue
                        obj = self._objective(candidate, move_cost)
                        if obj < old_obj and (choice is None or (obj, key) < (choice[0], choice[1])):
                            choice = (obj, key, candidate)
                        best_seen[key] = (obj, candidate)
            search_stats["passes"].append(stats)
            search_stats["total_generated_swaps"] += stats["generated_swaps"] + stats["fallback_generated_swaps"]
            search_stats["total_exact_swap_replays"] += stats["exact_swap_replays"]
            if choice is None:
                search_stats["no_move_reason"] = ("insufficient_confusion_evidence"
                                                  if not triggers and not stats["fallback_generated_swaps"]
                                                  else "no_exact_objective_improvement")
                break
            current = choice[2]
            seen.add(choice[1])
        baseline_key = tuple(self.baseline[sku] for sku in self.labels)
        chosen_key = tuple(current[sku] for sku in self.labels)
        selected_keys = {baseline_key, chosen_key}
        for key, _ in sorted(best_seen.items(), key=lambda x: (x[1][0], x[0])):
            if len(selected_keys) >= 5:
                break
            selected_keys.add(key)
        ranked = sorted(((key, best_seen[key]) for key in selected_keys),
                        key=lambda x: (x[1][0], x[0]))
        candidates = []
        for _, (obj, layout) in ranked:
            scored = self.score_current() if layout == self.baseline else self.score(layout)
            fixed = newly_wrong = 0
            for row in self.observations:
                oid, sku = row["id"], row["sku"]
                old_ok = baseline_score["predictions"][oid] == sku
                new_ok = scored["predictions"][oid] == sku
                fixed += not old_ok and new_ok
                newly_wrong += old_ok and not new_ok
            candidates.append({"objective": obj[0], "moved_skus": obj[1], "proxy_errors": obj[2],
                               "fixed": fixed, "new_errors": newly_wrong,
                               "layout": self._locations_for(layout)})
        recommended = next((c for c in candidates if c["layout"] == self._locations_for(current)), None)
        if recommended is None:
            scored = self.score_current() if current == self.baseline else self.score(current)
            fixed = newly_wrong = 0
            for row in self.observations:
                oid, sku = row["id"], row["sku"]
                old_ok = baseline_score["predictions"][oid] == sku
                new_ok = scored["predictions"][oid] == sku
                fixed += not old_ok and new_ok
                newly_wrong += old_ok and not new_ok
            recommended = {"objective": self._objective(current, move_cost)[0],
                           "moved_skus": sum(current[x] != self.baseline[x] for x in self.labels),
                           "proxy_errors": scored["errors"], "fixed": fixed,
                           "new_errors": newly_wrong, "layout": self._locations_for(current)}
        return {"step": self.data["step"], "snapshot_hash": self.data["snapshot_hash"],
                "model_hash": self.model.fingerprint, "baseline": {k: v for k, v in baseline_score.items() if k != "predictions"},
                "recommendation": recommended, "candidates": candidates,
                "transactions": self._transactions(current),
                "search": search_stats,
                "settings": {"move_cost": move_cost, "max_moved": max_moved, "cycle_pool_size": cycle_pool_size,
                             "swap_replay_limit": swap_replay_limit, "search_mode": search_mode,
                             "min_pair_segments": min_pair_segments, "min_pair_confusions": min_pair_confusions,
                             "min_pair_rate": min_pair_rate, "max_confusion_pairs": max_confusion_pairs,
                             "partners_per_shelf": partners_per_shelf,
                             "fallback_swap_budget": fallback_swap_budget}}

    def _locations_for(self, layout: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
        result = {sku: {"shelf": shelf, "slot": self.data["locations"][sku][1]}
                  for sku, shelf in layout.items() if shelf == self.baseline[sku]}
        free = {s: sorted({slot for sku, (old_shelf, slot) in self.data["locations"].items() if old_shelf == s}
                          - {x["slot"] for x in result.values() if x["shelf"] == s})
                for s in self.data["capacities"]}
        for sku in sorted(set(layout) - set(result)):
            shelf = layout[sku]
            preferred = self.data["locations"][sku][1]
            slot = preferred if preferred in free[shelf] else free[shelf][0]
            free[shelf].remove(slot)
            result[sku] = {"shelf": shelf, "slot": slot}
        return dict(sorted(result.items()))

    def _transactions(self, layout: Mapping[str, str]) -> List[Dict[str, str]]:
        locations = self._locations_for(layout)
        return [{"sku": sku, "from_shelf": self.baseline[sku],
                 "from_slot": self.data["locations"][sku][1],
                 "to_shelf": locations[sku]["shelf"], "to_slot": locations[sku]["slot"]}
                for sku in self.labels if layout[sku] != self.baseline[sku]]

    def diagnostics(self) -> Dict[str, Any]:
        baseline = self.score_current()
        per_sku: Dict[str, Dict[str, Any]] = {}
        evidence = self._confusion_evidence(self.baseline, baseline["predictions"])
        directed = {(row["a"], row["b"]): row["a_to_b"] for row in evidence}
        directed.update({(row["b"], row["a"]): row["b_to_a"] for row in evidence})
        for sku in self.labels:
            rows = [r for r in self.observations if r["sku"] == sku]
            own = np.concatenate([self.distances[r["id"]][:, self.label_index[sku]] for r in rows]) if rows else np.array([])
            wrong = sum(baseline["predictions"][r["id"]] != sku for r in rows)
            per_sku[sku] = {"segments": len(rows), "proxy_errors": wrong,
                            "own_distance_median": float(np.median(own)) if len(own) else None,
                            "own_distance_p90": float(np.percentile(own, 90)) if len(own) else None}
        similarity = self.model.centroids @ self.model.centroids.T
        pair_rows = []
        for i, source in enumerate(self.labels):
            for j, target in enumerate(self.labels):
                if i == j:
                    continue
                count = directed.get((source, target), 0)
                if count or similarity[i, j] > 0.9:
                    margins = [self.distances[r["id"]][:, j] - self.distances[r["id"]][:, i]
                               for r in self.observations if r["sku"] == source and len(self.distances[r["id"]])]
                    margin_values = np.concatenate(margins) if margins else np.array([])
                    pair_rows.append({"source": source, "target": target,
                                      "confusions": count, "source_segments": per_sku[source]["segments"],
                                      "directed_overlap_segments": float(self.directed_pair_risk[i, j]),
                                      "competitor_margin_p10": float(np.percentile(margin_values, 10)) if len(margin_values) else None,
                                      "centroid_cosine_distance": float(1 - similarity[i, j]),
                                      "same_shelf": self.baseline[source] == self.baseline[target]})
        pair_rows.sort(key=lambda r: (-r["confusions"], r["centroid_cosine_distance"], r["source"], r["target"]))
        return {"per_sku": per_sku, "confusing_pairs": pair_rows,
                "confusion_evidence": evidence,
                "baseline": {k: v for k, v in baseline.items() if k != "predictions"}}

    def retrospective(self, truth: Mapping[str, str], layout: Mapping[str, str]) -> Dict[str, Any]:
        """Post-decision analysis only; never called by recommend()."""
        expected_ids = {r["id"] for r in self.observations}
        if set(truth) != expected_ids or any(sku not in layout for sku in truth.values()):
            raise SnapshotError("Truth mapping must cover every observed ID with a known SKU")
        members = self._members(layout)
        errors = 0
        predictions = {}
        for row in self.observations:
            oid = row["id"]
            sku = truth[oid]
            shelf = row["observed_shelf"] if layout == self.baseline else layout[sku]
            pred = self._predict(oid, members[shelf])
            predictions[oid] = pred
            errors += pred != sku
        return {"errors": errors, "count": len(expected_ids),
                "accuracy": (len(expected_ids) - errors) / len(expected_ids) if expected_ids else None,
                "predictions": predictions}
