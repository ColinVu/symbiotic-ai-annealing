# Plan: shelf reassignment without inferred item labels

## Status and goal

This design is now implemented as the default `label_free` mode; `legacy` retains the prior inferred-label comparison. See [integration](INTEGRATION.md) for the chosen aggregation, explicit risk formula, and configurable uncalibrated settings, and [validation](LABEL_FREE_VALIDATION.md) for evidence. A deployed pick may provide an observed shelf and image embeddings but no item label. The new decision path should identify consistently ambiguous pairs from the classifier's candidate scores, propose capacity-preserving moves, and rank them without using inferred or true labels.

The simulator still owns whether a proposed move is applied. Verified labels, when available, belong in offline calibration and retrospective evaluation only.

## Proposed decision flow

1. **Score all candidates on the observed shelf.** Apply the saved embedding transforms and compute each frame's cosine distance to the centroids assigned to that shelf. Aggregate evidence across a segment for every shelf candidate; retain at least the two strongest candidates and their scores. The present weighted vote returns only a winner, so it does not supply enough information by itself. Define and calibrate one segment-level aggregation rule before implementation (for example, a mean of per-frame normalized distance scores); do not call uncalibrated distances probabilities.
2. **Detect a strong two-candidate ambiguity.** A segment supports pair `{A,B}` when A and B are its top two candidates, both have plausible absolute scores, and their score gap is small. A close gap between two poor candidates should not trigger a move. Count at most one vote per segment, aggregate repeated votes by unordered pair and observed shelf, and require minimum support across distinct observations. Tune the score and support thresholds on a separate labeled calibration set; the existing 5-segment / 2-error / 20% rule measures a different quantity and must not be carried over unchanged.
3. **Use centroid proximity as supporting evidence.** Compute cosine distance between the two centroids and, where available, reference-embedding spread. Close centroids increase concern but do not by themselves require a move: the pair must compete on the same shelf, appear ambiguous in observations, and have a safer destination. Keep a bounded geometry-based fallback for shelves with too little observation history.
4. **Generate and rank legal moves.** Reuse the capacity-preserving cross-shelf swap generator, slot assignment, move limit, and optional three-shelf cycles. Precompute a symmetric, label-free risk for each item pair and define layout risk as the sum for pairs sharing a shelf. Observed ambiguity can raise a pair's risk; centroid geometry supplies a risk estimate for pairs that have not yet competed on a shelf. For a proposed swap, update only the two affected shelves' risk rather than recomputing every layout from scratch. Include overlap newly introduced on both destination shelves and a cost for moved SKUs. Recommend a move only when this surrogate objective improves; otherwise return no move.
5. **Measure outcomes separately.** Report predicted ambiguity-risk reduction as a decision estimate, not errors fixed or accuracy gained. When verified labels or new post-move observations are available, evaluate true recognition errors and calibration separately. With only an observed shelf and no item/slot identity, an old pick cannot be replayed as though its unknown physical item had moved to a particular shelf.

## Code and interface changes

| Area | Current behavior | Proposed change |
| --- | --- | --- |
| `validate_snapshot()` | Requires `inferred_sku`; requires it to match the observed shelf. | Accept `observed_shelf` and embeddings without a label in the new mode. Preserve reveal-time, embedding, inventory, and shelf validation. An optional pickup slot may be recorded when the simulator can supply it. |
| `_predict()` / weighted vote | Returns only the winning SKU. | Expose per-segment candidate scores and a top-two ambiguity margin while retaining the existing winner for comparison. |
| `_confusion_evidence()` | Counts `inferred_sku -> predicted_sku` disagreements. | Replace its decision role with unordered top-two ambiguity evidence. Keep legacy confusion diagnostics only when weak labels are explicitly supplied. |
| `_pair_risk()` and `_swap_proxy_delta()` | Risk is relative to each observation's inferred SKU. | Build a symmetric, label-free pair-risk matrix from centroid proximity, reference spread where available, and observed ambiguity. Use it to rank exchange partners and penalize new destination-shelf overlap. |
| `score()`, `_objective()`, `_cycle_pool()` | Use inferred SKUs to replay proxy errors and select risky items. | Optimize layout ambiguity risk plus move cost; choose cycle seeds from ambiguous pairs. Do not assign old unlabeled segments to a proposed shelf by guessing their SKU. |
| CLI and reports | Require labels and report proxy errors, fixed errors, and new errors. | Add an explicit label-free mode and ambiguity settings. Report top-two evidence, pair support, destination risk, objective change, and why no move was chosen. Label legacy outputs clearly. |

Keep the present label-dependent path available for controlled comparison while the new path is validated. Separate their result schemas or add a clear metric type so a risk reduction cannot be mistaken for an accuracy gain.

## Calibration and deployment decisions remaining

- **Available pickup information:** confirm whether deployment knows only the pickup shelf or the exact bin/slot. Shelf-only operation must work without a guessed SKU; slot identity may permit stronger counterfactual evaluation.
- **Candidate evidence:** choose the segment score aggregation and calibrate what counts as two *strong* candidates. Test score-gap stability across lighting and video length.
- **Risk weights:** choose how observed ambiguity and centroid/reference overlap combine, and tune the move-cost scale against that new objective. The current `move_cost=0.25` is in proxy-error units and cannot be reused without recalibration.
- **Evaluation data:** identify a labeled set for offline threshold calibration and a separate set or new post-move recordings for outcome evaluation. Neither enters the decision snapshot.

## Verification and completion

1. A snapshot with no `inferred_sku` produces a valid analysis and recommendation; adding or changing an inferred or true label cannot change a label-free recommendation.
2. A clearly dominant candidate does not trigger a pair; two plausible close candidates do; two equally poor candidates do not.
3. Repeated ambiguity on a shared shelf nominates the pair, while one isolated ambiguous segment does not. Each segment contributes at most one vote.
4. A swap that separates A/B but creates comparable ambiguity with a destination occupant is rejected. All outputs preserve shelf capacity, unique slots, deterministic ordering, and the no-move option.
5. No future observations or retrospective labels influence the decision. Empty-frame segments are reported as unscorable rather than treated as ambiguity.
6. On held-out labeled and post-move data, compare ambiguity-risk changes with actual recognition errors; report misses, false alarms, run time, and moved SKUs. Do not claim accuracy gains from the surrogate objective alone.

**Completion criterion:** the new mode runs without item labels, emits reviewable ambiguity evidence and legal moves, passes the tests above, and has an evaluation report that clearly separates predicted risk from observed accuracy. Full-data and physical validation depend on complete model, snapshot, and recording artifacts.
