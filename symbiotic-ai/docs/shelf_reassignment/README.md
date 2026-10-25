# Shelf reassignment project

This directory documents the **decision component of a real-time shelf-change simulation**. Start with [EXECUTION_PLAN.md](EXECUTION_PLAN.md), use [EVALUATION_SPEC.md](EVALUATION_SPEC.md) for scoring and validation, and give [INTEGRATION.md](INTEGRATION.md) to the lab group building the bin-change simulator.

## Goal

Identify **when** the observed recognition evidence warrants changing a shelf layout, **which** objects to move, and **where** to put them without creating more confusion elsewhere. At each simulated decision point, recommend a capacity-preserving swap or cycle from information available so far. Another lab group owns the simulation of bin changes; this project supplies the decision logic and a clear interface for that simulator. It does not change physical inventory or live inference configuration.

## Research anchor

The supplied paper, *Symbiotic AI for Teaching a Computer to Identify Warehouse Objects Without Object Labels* (`/Users/nathanbai/Downloads/vu.pdf` at planning time), reports 96.9% weighted shelf-constrained segment accuracy over 120 objects with ten-fold cross-validation. Section 4.2 reports 94.6% for 38 visually similar “gems” versus 98.0% for other objects, including 10 of 25 `f14` segments mistaken for `f46`. The paper also identifies noisy `g32` embeddings. These are examples and investigation targets, not hard-coded move rules. The paper's 96.9% must not be compared directly with the checked-in subset's different five-fold results.

## Relevant code and data

| Existing component | Why it matters |
| --- | --- |
| `symbiote_weak_generalized/experiments/evaluator.py` | Canonical weighted shelf vote: nearest shelf centroid per frame, top-three distance margin weight, segment vote sum. Reuse or extract this logic. |
| `cross_validation/{layout,manifests,splits,test}.py` | Existing model/result artifacts and historical baseline. The shelf-change simulation does not require new held-out video testing. |
| `symbiote_weak_generalized/persistence/model_io.py` | Loads saved centroid arrays and optional embedding transforms. |
| `symbiote_weak_generalized/scripts/confusion_from_cache.py` | Useful cache-loading precedent, but its shelf-constrained segment mode is kNN, not the paper's weighted-centroid protocol. Do not use its confusion matrix as the optimization objective. |
| `ground_truth.csv` | Ordered true item labels for picklist videos. |

The checked-in `cv_results_subset` is a five-fold sample, not the full paper experiment. Its first fold has metadata for 119 centroids and no `centroids.npy`; the local picklist-video directory also contains no videos. Before running a complete simulation, obtain model artifacts, embeddings or videos/cache, labels, and inventory. Preserve the existing uncommitted work in `symbiote_weak_generalized/cli/main.py`, `core/config.py`, `experiments/assignment_score.py`, `experiments/evaluator.py`, plus the untracked ground-truth helper and test.

## Chosen defaults

- Five shelves, `c` through `g`, each with 24 object slots, unless an explicit inventory file overrides capacity.
- Immutable item IDs (for example `f14`) identify objects; the first character describes the **current** shelf only. Proposed shelves come from a separate layout map.
- Primary objective: reduce estimated weighted shelf-constrained **segment** errors at the current simulation state. Secondary objective: move fewer objects. Report per-item and per-shelf regressions.
- Use observed embeddings available at the decision point to account for within-class spread and between-class overlap. Centroid-to-centroid similarity is diagnostic, not sufficient to approve a move.
- Do not impose a fixed “similarity threshold” before seeing the data. A move is recommended when the currently observed evidence predicts a net gain after move cost. This is an in-simulation estimate, not a held-out accuracy claim. Swap search ranks all legal pairs by observed overlap, then exactly replays a configurable shortlist, so it does not prove global optimality.

## Completion definition

The implementation in `symbiotic-ai/shelf_layout/` consumes a simulation snapshot, reproduces the current-layout weighted-vote baseline, decides whether a move is warranted, generates legal layouts, and returns a proposed layout with estimated fixed/new errors. It includes tests for mapping, scoring, legality, destination-shelf regressions, and use of only observations available at the decision point. If required data are unavailable, it stops with an explicit missing-input report instead of inventing results. See [INTEGRATION.md](INTEGRATION.md) for the runnable interface.
