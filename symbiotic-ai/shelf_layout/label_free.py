"""Label-free centroid evidence and layout-risk optimization.

Distances and risk scores are surrogates, never calibrated probabilities.
"""
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
import math

import numpy as np


@dataclass(frozen=True)
class AmbiguitySettings:
    # Starting values for synthetic validation; calibrate on separate recordings.
    max_candidate_distance: float = 0.30
    max_score_gap: float = 0.03
    min_support: int = 3
    geometry_scale: float = 0.10
    observation_weight: float = 1.0
    fallback_min_risk: float = 0.5
    sparse_history_segments: int = 3

    def validate(self):
        values = asdict(self)
        if (any(not math.isfinite(v) or v < 0 for v in values.values())
                or self.geometry_scale <= 0 or self.max_candidate_distance > 2
                or self.max_score_gap > 2 or self.min_support < 2
                or not isinstance(self.min_support, int)
                or not isinstance(self.sparse_history_segments, int)):
            raise ValueError("Invalid ambiguity settings")


class LabelFreeMixin:
    def _label_free_risk(self):
        cfg = self.ambiguity
        centroids = self.model.centroids
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        geometry = np.clip(1 - centroids @ centroids.T, 0, 2)
        risk = np.exp(-geometry / cfg.geometry_scale)
        self.segment_evidence = []
        support = {}
        self.shelf_history = {s: 0 for s in self.data["capacities"]}
        members_by_shelf = self._members(self.baseline)
        for row in self.observations:
            shelf = row["observed_shelf"]
            members = members_by_shelf[shelf]
            d = self.distances[row["id"]]
            evidence = {"id": row["id"], "observed_shelf": shelf, "scorable": bool(len(d)),
                        "aggregation": "mean_frame_cosine_distance", "candidates": [],
                        "ambiguous": False, "score_gap": None,
                        "weighted_vote_winner": self._predict(row["id"], members)}
            if len(d):
                self.shelf_history[shelf] += 1
                scores = sorted((float(d[:, self.label_index[x]].mean()), x) for x in members)
                evidence["candidates"] = [{"sku": x, "distance": score} for score, x in scores]
                gap = scores[1][0] - scores[0][0]
                evidence["score_gap"] = gap
                evidence["ambiguous"] = scores[1][0] <= cfg.max_candidate_distance and gap <= cfg.max_score_gap
                if evidence["ambiguous"]:
                    a, b = sorted((scores[0][1], scores[1][1]))
                    support.setdefault((shelf, a, b), []).append(row["id"])
            self.segment_evidence.append(evidence)
        self.ambiguity_pairs = []
        for (shelf, a, b), ids in sorted(support.items()):
            i, j = self.label_index[a], self.label_index[b]
            eligible = len(ids) >= cfg.min_support
            # Isolated ambiguous segments cannot increase decision risk.
            boost = cfg.observation_weight * len(ids) / self.shelf_history[shelf] if eligible else 0.0
            risk[i, j] += boost
            risk[j, i] += boost
            self.ambiguity_pairs.append({"a": a, "b": b, "observed_shelf": shelf,
                                         "support": len(ids), "observation_ids": ids,
                                         "centroid_cosine_distance": float(geometry[i, j]),
                                         "eligible": eligible,
                                         "exclusion_reasons": [] if eligible else ["insufficient_support"]})
        np.fill_diagonal(risk, 0)
        return risk

    def _risk_score(self, layout):
        if set(layout) != set(self.labels) or any(s not in self.data["capacities"] for s in layout.values()):
            raise ValueError("Layout must assign every SKU to a known shelf")
        members = self._members(layout)
        if any(len(members[s]) != cap for s, cap in self.data["capacities"].items()):
            raise ValueError("Layout violates shelf capacities")
        shelves = {s: sum(float(self.pair_risk[self.label_index[a], self.label_index[b]])
                          for a, b in combinations(group, 2)) for s, group in members.items()}
        return {"metric_type": "ambiguity_risk", "ambiguity_risk": sum(shelves.values()),
                "per_shelf_risk": shelves}

    def _label_free_diagnostics(self):
        return {"baseline": self.score_current(), "segment_evidence": self.segment_evidence,
                "ambiguity_evidence": self.ambiguity_pairs,
                "reference_spread": "unavailable_in_saved_model_contract",
                "aggregation": "mean_frame_cosine_distance",
                "ambiguity_settings": asdict(self.ambiguity),
                "calibration_status": "uncalibrated",
                "unscorable_segments": [r["id"] for r in self.segment_evidence if not r["scorable"]]}

    def _recommend_label_free(self, *, move_cost, max_moved, cycle_pool_size,
                              swap_replay_limit, search_mode, max_pairs,
                              partners_per_shelf, fallback_swap_budget):
        if (not math.isfinite(move_cost) or move_cost < 0 or max_moved < 0
                or cycle_pool_size < 0 or swap_replay_limit < 1 or max_pairs < 1
                or partners_per_shelf < 1 or fallback_swap_budget < 0
                or search_mode not in {"confusion", "all_pairs"}):
            raise ValueError("Invalid search settings")
        current = dict(self.baseline)
        base = self._risk_score(current)
        def key(layout):
            return tuple(layout[x] for x in self.labels)
        def objective(layout, risk):
            moved = sum(layout[x] != self.baseline[x] for x in self.labels)
            return (risk + move_cost * moved, moved, risk)
        initial = objective(current, base["ambiguity_risk"])
        visited = {key(current): (initial, dict(current))}
        passes = []
        while True:
            old = visited[key(current)][0]
            totals = self._risk_totals(current)
            triggers = [r for r in self.ambiguity_pairs if r["eligible"] and current[r["a"]] == current[r["b"]]]
            triggers.sort(key=lambda r: (-r["support"], r["a"], r["b"]))
            pairs = [(r["a"], r["b"]) for r in triggers[:max_pairs]]
            # Geometry fallback only on shelves with insufficient usable history.
            fallback = [(a, b) for a, b in self._fallback_pairs(current, len(self.labels) ** 2)
                        if self.shelf_history[current[a]] < self.ambiguity.sparse_history_segments
                        and self.pair_risk[self.label_index[a], self.label_index[b]] >= self.ambiguity.fallback_min_risk]
            fallback = fallback[:fallback_swap_budget]
            proposals = self._targeted_swaps(current, pairs + fallback, partners_per_shelf, max_moved, totals)
            if search_mode == "all_pairs":
                proposals = [(self._swap_proxy_delta(current, a, b, totals), a, b, (a, b))
                             for a, b in combinations(self.labels, 2) if current[a] != current[b]]
                proposals.sort()
            stats = {"trigger_pairs": triggers, "fallback_pairs": fallback,
                     "generated_swaps": len(proposals), "evaluated_swaps": [], "evaluated_cycles": []}
            choice = None
            for delta, a, b, pair in proposals[:swap_replay_limit]:
                candidate = dict(current)
                candidate[a], candidate[b] = candidate[b], candidate[a]
                k = key(candidate)
                obj = objective(candidate, max(0.0, old[2] + delta))
                if k in visited or obj[1] > max_moved:
                    continue
                visited[k] = (obj, candidate)
                stats["evaluated_swaps"].append({"a": a, "b": b, "trigger_pair": pair,
                                                   "risk_delta": delta, "objective": obj[0],
                                                   "destination_shelf_risk": self._affected_risk(candidate, (current[a], current[b]))})
                if obj[0] < old[0] - 1e-12 and (choice is None or (obj, k) < choice[:2]):
                    choice = (obj, k, candidate)
            if choice is None and cycle_pool_size >= 3 and pairs:
                seeds = list(dict.fromkeys(x for pair in pairs for x in pair))
                ranked = sorted(self.labels, key=lambda x: (-float(totals[current[x]][self.label_index[x]]), x))
                pool = list(dict.fromkeys(seeds + ranked))[:cycle_pool_size]
                for triple in combinations(pool, 3):
                    if len({current[x] for x in triple}) != 3:
                        continue
                    for rotated in permutations([current[x] for x in triple]):
                        if any(rotated[i] == current[x] for i, x in enumerate(triple)):
                            continue
                        candidate = dict(current)
                        candidate.update(zip(triple, rotated))
                        k = key(candidate)
                        if k in visited:
                            continue
                        obj = objective(candidate, self._risk_score(candidate)["ambiguity_risk"])
                        if obj[1] > max_moved:
                            continue
                        visited[k] = (obj, candidate)
                        stats["evaluated_cycles"].append({"skus": list(triple),
                                                          "destination_shelves": list(rotated),
                                                          "risk_delta": obj[2] - old[2],
                                                          "objective": obj[0]})
                        if obj[0] < old[0] - 1e-12 and (choice is None or (obj, k) < choice[:2]):
                            choice = (obj, k, candidate)
            passes.append(stats)
            if choice is None:
                break
            current = choice[2]
        def describe(obj, layout):
            return {"metric_type": "ambiguity_risk", "objective": obj[0], "moved_skus": obj[1],
                    "ambiguity_risk": obj[2], "predicted_risk_reduction": initial[2] - obj[2],
                    "objective_change": obj[0] - initial[0],
                    "per_shelf_risk": self._risk_score(layout)["per_shelf_risk"],
                    "layout": self._locations_for(layout)}
        ranked = sorted(visited.values(), key=lambda x: (x[0], key(x[1])))
        selected = ranked[:5]
        if not any(layout == self.baseline for _, layout in selected):
            selected = selected[:4] + [(initial, self.baseline)]
        return {"metric_type": "ambiguity_risk", "step": self.data["step"],
                "snapshot_hash": self.data["snapshot_hash"], "model_hash": self.model.fingerprint,
                "baseline": base, "recommendation": describe(*visited[key(current)]),
                "candidates": [describe(*v) for v in selected], "transactions": self._transactions(current),
                "evidence": self._label_free_diagnostics(),
                "search": {"passes": passes, "no_move_reason": (
                    ("insufficient_ambiguity_evidence" if search_mode == "confusion"
                     and not passes[0]["trigger_pairs"] and not passes[0]["fallback_pairs"]
                     else "no_surrogate_objective_improvement") if current == self.baseline else None)},
                "settings": {"mode": "label_free", "move_cost": move_cost, "max_moved": max_moved,
                             "cycle_pool_size": cycle_pool_size, "swap_replay_limit": swap_replay_limit,
                             "search_mode": search_mode, "max_pairs": max_pairs,
                             "partners_per_shelf": partners_per_shelf, "fallback_swap_budget": fallback_swap_budget,
                             "ambiguity": asdict(self.ambiguity), "calibration_status": "uncalibrated"}}

    def _affected_risk(self, layout, shelves):
        return {s: sum(float(self.pair_risk[self.label_index[a], self.label_index[b]])
                       for a, b in combinations([x for x in self.labels if layout[x] == s], 2))
                for s in shelves}
