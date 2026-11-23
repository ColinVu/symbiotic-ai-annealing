# Simulator integration contract

The implementation is in `symbiotic-ai/shelf_layout/`. The bin-change simulator supplies a complete snapshot at each decision point and owns whether or how a recommendation is applied. The engine never mutates its input or applies a physical move.

## Default label-free API

```python
from shelf_layout import AmbiguitySettings, DecisionEngine, load_model

model = load_model("/path/to/model")
engine = DecisionEngine(snapshot, model, ambiguity=AmbiguitySettings(
    max_candidate_distance=0.30, max_score_gap=0.03, min_support=3))
decision = engine.recommend(max_moved=12)  # default move cost: 0.02 risk units/SKU
```

Supply `step`, `capacities`, complete `items` (SKU, shelf, slot), and `observations` with `id`, `revealed_at`, `observed_shelf`, and raw `frames`. No `inferred_sku` is required; if supplied, it and assignment confidence are ignored, including in the decision fingerprint. Ground-truth fields are rejected. Optional `pickup_slot` must name an occupied slot on the observed shelf; it is retained as audit metadata and does not influence decisions. All reveal-time, capacity, inventory, embedding, and transform validation still applies. Empty-frame segments are unscorable and cast no vote.

CLI commands default to `--mode label_free`; use `--mode legacy` for comparison. All commands accept `--max-candidate-distance` (0.30), `--max-score-gap` (0.03), `--min-ambiguity-support` (3), `--geometry-scale` (0.10), `--observation-weight` (1), `--fallback-min-risk` (0.5), and `--sparse-history-segments` (3). These are **uncalibrated starting settings**, not production thresholds. `--move-cost` defaults to 0.02 in label-free mode and 0.25 in legacy mode. Legacy confusion thresholds apply only to legacy decisions. `--max-confusion-pairs` bounds nominated pairs in either mode. `--search-mode all_pairs` offers a broad geometry-risk comparison, bypassing targeted ambiguity gates.

Segment scores are mean frame cosine distances to every candidate on the observed shelf, not probabilities. A segment votes only if its second-best distance is at most the strength threshold and its top-two gap is at most the gap threshold. For centroid distance `d`, geometry risk is `exp(-d / geometry_scale)`. A pair meeting minimum support gets an additional `observation_weight * support / usable_segments_on_observed_shelf`. The symmetric pair matrix is fixed for the snapshot; layout risk sums it once per same-shelf unordered pair. No reference spread is used because the saved model contract contains no reference embeddings; reports say so explicitly.

The recommendation contains `ambiguity_risk`, `predicted_risk_reduction`, `objective_change`, per-shelf destination risks, a complete layout, and transactions. `evidence` contains segment scores, top-two gaps, weighted-vote winners for comparison, pair support, eligibility, and unscorable IDs. Search passes record nominated/fallback pairs, evaluated swap deltas, and affected-shelf risks. `summary.json`, `segment_evidence.json`, `ambiguity_evidence.csv`, `candidate_layouts.csv`, `move_transactions.csv`, and `decisions.csv` are separate from legacy proxy-error reports. `analyze` reports evidence and current risk without searching.

The engine does not know an old shelf-only pick's identity after a move. `--truth-json` is read after selection and reports only current-layout retrospective accuracy in label-free mode; proposed accuracy requires new post-move snapshots. `simulate` respects each supplied inventory and never applies a proposal itself. See [validation](LABEL_FREE_VALIDATION.md).

## Legacy API (`mode="legacy"`, CLI `--mode legacy`)

The rest of this guide documents the retained inferred-label comparison path.

## Snapshot shape

```json
{
  "step": 42,
  "capacities": {"c": 24, "d": 24, "e": 24, "f": 24, "g": 24},
  "items": [
    {"sku": "c11", "shelf": "c", "slot": "11"}
  ],
  "observations": [
    {
      "id": "picklist_101:0",
      "revealed_at": 40,
      "inferred_sku": "c11",
      "observed_shelf": "c",
      "assignment_confidence": 0.83,
      "frames": [[0.1, 0.2]]
    }
  ],
  "previous_decision": {"step": 41, "accepted": true}
}
```

The example is abbreviated: `items` must include every SKU and slot, and each frame must have the saved model's embedding dimension (usually 512). `frames` are raw cached CLIP embeddings; the engine applies saved hand neutralization and adapter transforms. `inferred_sku` is the weak assignment known at that step, not ground truth. `observed_shelf` is the pickup shelf **in the simulator's current snapshot**; current-layout classification uses it directly. For a new pick it comes from the pickup event or marker. If an old recorded pick is replayed after a simulated move, the simulator must remap its pickup shelf to the current layout; the original marker remains historical data. The inferred SKU must currently belong to the supplied shelf, or the engine rejects the observation rather than treating an inconsistent assignment as evidence for a move. An observation may have an empty `frames` list; it counts as an unfixable proxy error. `previous_decision` is optional audit metadata. No observation may have `revealed_at > step`. Do not include `ground_truth`, `true_label`, `truth`, or `expected_label` in a snapshot.

The snapshot's `items` describe the simulator's **current** layout. If a prior proposal was rejected, keep the old layout; if accepted, supply the changed layout and pickup events with their observed shelves. The engine does not infer acceptance from its earlier output. Candidate replay assumes the weakly assigned item is physically picked from its **proposed** shelf; both the item and its catalog centroid move together in this counterfactual. This is an estimate from old images, not a simulated post-move camera recording.

## Python API

```python
from shelf_layout import DecisionEngine, load_model

model = load_model("/path/to/saved/model")
engine = DecisionEngine(snapshot, model, mode="legacy")
decision = engine.recommend(move_cost=0.25, max_moved=12)
```

`decision["recommendation"]["layout"]` is a complete `sku -> {shelf, slot}` map. `decision["transactions"]` lists the SKUs whose shelf changes. Empty transactions mean “do not move yet.” The result also includes the current proxy error count, estimated fixed and new errors, five ranked candidate layouts, snapshot/model hashes, search settings, and per-pass search evidence. By default, the optimizer nominates same-shelf pairs with repeated weighted-vote confusion, ranks bounded cross-shelf exchange partners, and exactly replays the shortlist. A bounded overlap fallback handles sparse evidence; three-shelf cycles remain available. `search_mode="all_pairs"` retains the previous strategy for comparisons. `swap_replay_limit`, evidence thresholds, partner count, fallback budget, and `cycle_pool_size` are tunable speed/coverage controls; the search is heuristic rather than a proof of global optimality.

The default trigger requires at least five valid segments assigned to the two items combined, at least two cross-predictions between them, and a combined confusion rate of at least 20%; the items must currently share a shelf. These are configurable starting thresholds. `confusion_evidence.csv` lists pair counts, rates, eligibility, and reasons a pair did not qualify. `summary.json` includes `search.passes` with trigger pairs, candidate provenance, replay counts, and fallback use. The CLI exposes these controls as `--min-pair-segments`, `--min-pair-confusions`, `--min-pair-rate`, `--max-confusion-pairs`, `--partners-per-shelf`, and `--fallback-swap-budget`.

## CLI

Run from `symbiotic-ai/` after installing its package or setting `PYTHONPATH`:

```bash
python -m shelf_layout preflight --snapshot snapshot.json --model-dir model --output-dir report
python -m shelf_layout analyze --snapshot snapshot.json --model-dir model --output-dir report
python -m shelf_layout recommend --snapshot snapshot.json --model-dir model --output-dir report
python -m shelf_layout simulate --snapshots-dir snapshots --model-dir model --output-dir timeline
```

`simulate` reads all JSON snapshots from the supplied directory in ascending `step` order. It calls the decision engine independently for each snapshot, writes a step report, and writes a top-level `decisions.csv`. It does not change layouts between snapshots; the simulator must supply each next layout. A failed preflight writes `missing_inputs.json` and returns exit code 2.

For retrospective analysis, `recommend` accepts `--truth-json path`, where the file maps observation IDs to true SKUs. The CLI loads this file **after** the recommendation is fixed and writes separate true-label error counts. It never affects move selection. The primary reports remain in-simulation estimates from existing observations, not results on new videos.

## Required data before a full study run

The checked-in five-fold subset is incomplete for this feature: its first model lists 119 centroids but lacks `centroids.npy`, and the local picklist-video directory is empty. Obtain a complete saved model, all SKU locations, and the cached segment embeddings or simulator snapshots. The engine can be developed and tested with synthetic snapshots in the meantime; it must not report a numerical gain for the paper dataset until the full inputs are supplied.

## Label-free handoff

The default decision snapshot retains `step`, capacities, complete item-to-shelf/slot inventory, observed pickup shelf, and frame embeddings. It does **not require `inferred_sku`**; a pickup slot may be included if the simulator actually knows it. The engine returns top-two ambiguity evidence, estimated layout-risk change, a complete proposed layout, and transactions. `proxy_errors`, `fixed`, and `new_errors` remain legacy or retrospective metrics, not decision-time claims for unlabeled picks. The simulator must supply its actual next layout after accepting or rejecting a proposal.
