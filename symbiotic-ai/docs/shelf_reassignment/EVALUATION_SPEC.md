# Simulation scoring, tests, and acceptance contract

## Definitions

- **SKU:** immutable object ID, for example `f14`.
- **Layout:** total map from SKU to shelf and physical slot. Each shelf has its declared capacity.
- **Scored segment:** one ground-truth carry segment with a true SKU and a corresponding centroid. Segments without valid frame embeddings remain in the accuracy denominator as failures and cannot be repaired by relocation.
- **Decision-time proxy error:** weighted shelf vote differs from the revealed weak SKU assignment. This is available to the move recommender and may be wrong.
- **Retrospective error:** weighted shelf vote differs from the true SKU in `ground_truth.csv`; compute only in the analysis layer after a simulation decision.
- **Fixed/new errors:** segments wrong under the current layout and right under the candidate / right under the current layout and wrong under the candidate, measured separately for proxy and retrospective labels. Net gain equals fixed minus new errors.

## Exact score and explanatory statistics

For frame embedding `x`, centroid `c_j`, and shelf `S`, compute `d_j = 1 - cos(x,c_j)` for `j in S`. For the current layout, `S` is the observed pickup shelf. For a proposed layout, assume the weakly inferred SKU is physically moved and picked from its proposed shelf; never use a hidden true label to choose `S` during a decision. Sort by `(distance, SKU)` and let `j1,j2,j3` be the three nearest. Put `m12=max(0,d2-d1)` and `m13=max(0,d3-d1)`. Match `_harmonic_three_centroid_margin` exactly: return `0` if both gaps are zero, `m13` if only `m12` is zero, `m12` if only `m13` is zero, otherwise `2*m12*m13/(m12+m13)`. Add this weight to `j1`'s segment total. Predict the SKU with the largest total, with the existing evaluator's deterministic tie rule. Score one correct/incorrect result per segment, not per frame. In retrospective analysis only, use the true SKU's candidate shelf to compare with paper-style evaluation and label that metric separately.

For each inferred class A, report own-centroid distance median and 90th percentile; number of segments; and competitor margin distribution for candidate B. Report directed `A -> B` segment counts and denominators, never just cosine similarity. A pair with low centroid distance but no actual confusion is explanatory context, not automatically a relocation target. The score must consider all competing SKUs on a shelf simultaneously. True-label versions belong only in retrospective reports.

Search objective on observations revealed at the current simulation step: `J(layout) = number_of_proxy_segment_errors(layout) + lambda * number_of_SKUs_whose_shelf_differs_from_current_layout`. Default `lambda=0.25`, default maximum moved SKUs per decision `12`. Lower is better; equal objective favors fewer moves, then lexical SKU order. The optimizer should also expose `lambda=0` and move-cap overrides for sensitivity analysis. Report raw proxy error counts separately from `J`, along with retrospective true-label error counts when available. The latter never influences the search.

## Required tests

1. **Parity:** current-layout replay gives the same weighted prediction for every covered segment as `experiments/evaluator.py`, including top-three margin and ties.
2. **Layout identity:** `f14` remains `f14` after relocation; candidate shelf filtering uses the explicit map and no SKU prefix or video suffix.
3. **Capacity:** swaps and three-way cycles preserve one instance per SKU, 24 objects per study shelf, and unique slots; illegal maps fail validation.
4. **Regression accounting:** construct a synthetic example where separating A/B fixes an error but moving A near C creates a new error. Verify both appear and net gain is correct.
5. **Intraclass effect:** a high-variance class can remain unreliable even when its nearest old competitor is removed; diagnostic spread and replay reflect this.
6. **No future or label oracle:** a decision at step `t` cannot read observations revealed after `t` or `ground_truth.csv`; changing future embeddings or hidden true labels does not change that step's proposal.
7. **Determinism:** identical inputs, seed, and parameters yield byte-stable layout and move tables.
8. **Data failures:** missing centroid/model file, malformed inventory, duplicate SKU/slot, dimension mismatch, or malformed observations produce clear preflight failures without partial recommendations. Ground truth is optional and separate from the decision input.
9. **No-move choice:** when every legal change has nonpositive estimated net benefit from revealed observations, recommend zero moves.
10. **Simulator handoff:** return a complete proposed layout and transaction list without mutating the input snapshot; the next call respects whether the simulator accepted or rejected the prior proposal.

## Acceptance criteria

- Complete inputs pass preflight and the baseline scorer matches the canonical evaluator on every covered segment, with identical denominator handling.
- Each emitted candidate is a legal full 120-SKU assignment with a reviewable sequence of transactions and explicit fixed/new error counts.
- The final report shows a decision timeline, predicted proxy improvement at each step, and retrospective true-label results where available. It makes no held-out or new-video accuracy claim.
- `f14`/`f46` and `g32` appear in diagnostics if their input data are present, but no test requires that a specific object move; the decision follows measured net benefit.
- Incomplete checked-in artifacts produce `missing_inputs.json` and no claimed accuracy gain.

## Interpretation limit

Replay holds visual embeddings, centroids, and action segmentation fixed. Moving products may change backgrounds, hand pose, lighting, pick order, and marker visibility; the decision component estimates recognition changes from the shelf candidate set only. The other lab group's simulator owns bin-change effects and timing. The reported numbers describe this simulation on existing videos and do not establish how the system will perform on new video or after physical relocation.
