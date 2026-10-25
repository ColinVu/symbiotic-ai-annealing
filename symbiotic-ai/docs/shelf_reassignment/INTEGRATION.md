# Simulator integration contract

The implementation is in `symbiotic-ai/shelf_layout/`. The bin-change simulator supplies a complete snapshot at each decision point and owns whether or how a recommendation is applied. The engine never mutates its input or applies a physical move.

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
engine = DecisionEngine(snapshot, model)
decision = engine.recommend(move_cost=0.25, max_moved=12)
```

`decision["recommendation"]["layout"]` is a complete `sku -> {shelf, slot}` map. `decision["transactions"]` lists the SKUs whose shelf changes. Empty transactions mean “do not move yet.” The result also includes the current proxy error count, estimated fixed and new errors, five ranked candidate layouts, snapshot/model hashes, and search settings. The optimizer ranks all cross-shelf swaps by observed pairwise overlap, exactly replays the top 200 by default, and considers three-shelf cycles among high-risk SKUs. `swap_replay_limit` and `cycle_pool_size` are tunable speed/coverage controls; the search is heuristic rather than a proof of global optimality.

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
