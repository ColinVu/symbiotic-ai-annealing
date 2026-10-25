# Execution plan: shelf decisions in a real-time bin-change simulation

## 1. Intended result and boundaries

Build a decision component that answers, at each simulation step: **should any object move now, which objects, to which shelves, and what net change in recognition errors is expected?** The other lab group owns applying bin changes in the simulator. This component consumes its current state and returns a ranked move proposal, including the risk of new confusion on destination shelves. It never applies a physical or live-system change.

Treat an object label as an immutable SKU key. Its existing prefix is only the original location. The current or candidate location is `layout[sku]`; never infer the current shelf from `sku[0]` or the picklist-stem suffix. Current-layout classification uses the pickup event's observed shelf. During decision-time counterfactual replay, move the inferred SKU and its reference centroid together, and use that SKU's proposed shelf as the assumed new pickup shelf. Never use a hidden ground-truth SKU to choose it. This changes the classifier's candidate set while holding the observed image fixed; the other group's simulator is responsible for representing bin changes and any associated scene effects.

## 2. Inputs and preflight gate

Add a `shelf_layout` package under `symbiotic-ai/` with a callable decision API and a CLI harness. A decision snapshot contains: current step ID/time, current `sku -> (shelf,slot)` layout and capacities, the saved centroid model and transforms, and only segment embeddings, inferred SKU assignments, and observed pickup shelves that the simulator has revealed by that step. A separate analysis input may include ordered true labels from `ground_truth.csv` for scoring after decisions; the decision API must not read those labels. The CLI accepts snapshot/model paths and an output directory. Record hashes of model, configuration, inventory, and revealed-observation set in the result.

Preflight must verify: 120 unique SKUs in the five-by-24 study setup; each SKU maps to one shelf and slot; capacities are respected; every candidate SKU has a centroid; embeddings are finite, nonempty, and in the expected dimension; each observation has a reveal step no later than the current step; no future observation or evaluation-only true label enters the decision input. Report per-class sample counts and missing/invalid data. Stop if a required centroid or snapshot input is missing; do not silently remove a class or fall back to unconstrained classification. The checked-in subset cannot pass this gate as-is.

## 3. Build an exact counterfactual scorer

Extract or share the weighted shelf-vote implementation in `symbiote_weak_generalized/experiments/evaluator.py` so baseline and hypothetical layouts use identical behavior: for each frame, rank only centroids assigned to the candidate shelf; vote for the nearest; weight that vote by the harmonic mean of the first-to-second and first-to-third cosine-distance gaps; sum weights across the segment; resolve label ties deterministically. Preserve the existing behavior for zero-weight votes and missing frames. A segment with no valid frames is a counted failure, as in the evaluator, and is marked **unfixable by layout**.

Precompute the transformed frame-by-centroid distance matrix for the observations revealed in the current snapshot, with stable SKU ordering. A candidate layout changes only the shelf mask, so no CLIP rerun or centroid retraining is required for each candidate. First replay the snapshot's current layout and require exact per-segment prediction and aggregate metric agreement with the canonical evaluator on the same inputs. If it disagrees, fix the data or scorer before optimizing.

For diagnostics, calculate each class's distribution of own-centroid distances (intraclass spread), pairwise centroid distances (interclass closeness), and directed counterfactual confusion `A -> B` when A and B compete on a shelf. Use counts and rates with denominators; distinguish a consistently overlapping pair from a noisy item such as `g32`. These diagnostics explain proposals; the actual weighted-vote replay determines their score. If weak assignments are uncertain, include their confidence and mark the affected estimates as provisional.

## 4. Search legal layouts

Represent a layout as `sku -> shelf` with immutable SKU keys and exactly 24 occupants per shelf. Start from the simulator's current map. Enumerate all legal cross-shelf swaps, rank them by observed pairwise embedding overlap, and exactly replay the top 200 by default; each swap moves two objects. Use deterministic best-improvement local search among replayed swaps: accept the best legal swap only if it improves `estimated_errors + lambda * moved_skus`, and continue until no swap improves it or the configured move limit is reached. Expose the exact-replay limit, `lambda = 0.25` segment errors per moved SKU, and move cap as CLI parameters. The pairwise rank is only a speed filter; exact weighted-vote replay decides whether a move is beneficial. Evaluate the objective from the **snapshot's current** layout, not the last swap alone. Reject repeated layouts and use a fixed SKU sort order for ties.

After swap search, evaluate three-object cycles involving three distinct shelves from the resulting layout. Accept the best improving cycle and repeat swap-plus-cycle passes until no improvement. Cap one decision at 12 moved SKUs by default, measured against the snapshot's current layout; expose this cap. Never return a layout that exceeds capacity or whose objective is worse than staying put. Include the no-move option in every ranked result. Preserve the best five distinct legal layouts for the report rather than only the last visited layout.

For each recommendation, produce a transaction list of slot-level swaps/cycles. Reuse each moved object's source slot on its destination shelf only when unoccupied after the simultaneous transaction; otherwise assign the lowest available slot in row-major order. Validate the complete final SKU-to-slot map before emitting it. The simulator may accept or reject the proposal and owns the actual state transition.

## 5. Run the existing-video simulation

The other group supplies a sequence of bin-layout snapshots and applies changes. At each decision point, consume only the observations revealed up to that point, calculate the best no-move/swap/cycle options, and return the proposal and its evidence. Save the decision step, input snapshot hash, proposed moves, predicted fixed/new errors, and acceptance/rejection returned by the simulator. On the next step, read the simulator's updated layout; do not assume a proposal was applied.

Use the existing videos for the simulation and its counterfactual replay. The simulation may use the same observations to select and score swaps; state this plainly as an **in-simulation estimate**, not a held-out or future-video generalization result. Ground-truth labels, where available, are used only for retrospective analysis after each decision, not as an oracle for the decision logic. Report when each move was proposed/applied, current versus simulated accuracy and error counts, fixed and newly introduced errors, per-shelf/per-class changes, and the number of moved objects. Do not quote the paper's 96.9% as the baseline for a different model or simulation.

The first integration can use a static snapshot from the existing data while the other group's simulator interface is still being finalized. Keep the decision API independent of that harness so the simulator can call it at each step without reimplementing scoring. If no move has positive estimated net value at a step, return an explicit no-move decision and wait for more observations.

## 6. Outputs and user workflow

Emit `summary.md`, `summary.json`, `decisions.csv`, `candidate_layouts.csv`, `move_transactions.csv`, `per_sku_metrics.csv`, `per_shelf_metrics.csv`, `confusing_pairs.csv`, and `missing_inputs.json` when preflight fails. The summary states the simulation timeline, current and proposed error counts and accuracy, number of moved SKUs, errors fixed/newly introduced, observations available at each step, move-cost settings, and limitations of replaying old camera data under a changed layout. Include paths to generated tables and the exact command/configuration used.

CLI subcommands: `preflight`, `analyze`, and `recommend`. `preflight` validates a snapshot; `analyze` writes current-layout diagnostics; `recommend` returns a no-move or layout-change proposal. Expose the same operations as importable Python functions for the other group's simulator. Use explicit input paths and output directory, with a fixed random seed. Do not edit the existing CV artifact tree.

## 7. Implementation order

1. Inventory schema, data adapters, and strict preflight. Confirm complete model and cache artifacts are accessible before expensive work.
2. Shared weighted-vote scorer and exact baseline parity against the existing evaluator.
3. Intraclass/interclass diagnostics and directed error attribution.
4. Capacity-preserving search with deterministic move-cost rules and transaction generation.
5. Snapshot API and static existing-video simulation harness; coordinate the exact adapter with the group building bin-change simulation.
6. Timeline reports, examples, and tests described in `EVALUATION_SPEC.md`.

The implementation should not modify existing uncommitted files without incorporating their current contents. Extend the current evaluator only as needed to share scoring; avoid routing the optimization through `scripts/confusion_from_cache.py` because its segment-level shelf mode uses kNN.
