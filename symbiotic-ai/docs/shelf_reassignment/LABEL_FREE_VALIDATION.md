# Label-free implementation validation

Validated on November 9, 2026. The default Python and CLI paths use centroid distances and observed top-two ambiguity. The inferred-label comparison requires explicit `mode="legacy"` / `--mode legacy`.

## Automated verification

Run from the repository root with Python and NumPy available:

```bash
PYTHONPATH=symbiotic-ai python3 -m unittest discover -s symbiotic-ai/tests -p 'test_shelf_layout*.py'
```

24 focused tests pass (14 retained legacy tests and 10 label-free tests). They cover unlabeled snapshots; invariance to missing, changed, invalid inferred labels and assignment confidence; rejection of future observations and truth fields; saved transforms; segment-level support independent of frame count; dominant and equally poor candidates; minimum support; sparse-history fallback; unscorable empty segments; deterministic legal slots and capacities; the no-move and move-cap constraints; destination overlap; incremental swap deltas matching full-layout risk; a three-shelf cycle that improves when no swap can; CLI reports; separate retrospective truth; and simulator timeline output.

Label-free candidate layouts never replay old shelf-only observations. The retrospective API rejects proposed-layout scoring in this mode; the CLI reports only baseline true-label accuracy after the recommendation is fixed. Candidate outcome accuracy is null and requires new observations.

## Larger synthetic check

[Raw results and fixture definition](LABEL_FREE_SYNTHETIC_RESULTS.json) describe a deterministic 120-item, five-shelf fixture with one close centroid pair and three unlabeled ambiguous observations. All other centroids are orthogonal. Settings use defaults with a two-SKU move cap and cycles disabled.

| Search | Generated swaps across passes | Evaluated swaps | Baseline risk | Proposed risk | Moved SKUs | Local runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Targeted ambiguity | 40 | 40 | 2.013239 | 0.062652 | 2 | 0.0092 s |
| All pairs | 11,520 | 200 | 2.013239 | 0.062652 | 2 | 0.0293 s |

Both searches separate the close pair and preserve all shelf capacities and slots. The all-pairs count includes the final search pass; its scoring shortlist remains bounded. Runtime is a single local measurement, not a performance guarantee. These values are **surrogate risk**, not recognition errors, probabilities, or measured accuracy. This favorable fixture does not establish search quality on real recordings.

## Validation still required

The absolute-distance threshold, gap, support count, geometry scale, observation weight, and movement cost need calibration on separate labeled data. Reference spread is unavailable in the saved model contract, so the implementation explicitly reports that it uses centroid geometry without reference spread. The current saved-model subset is missing centroid arrays and full recordings; no held-out accuracy, false-alarm, miss-rate, or physical outcome claim is made here. Complete models, independent labeled calibration/evaluation sets, and post-move recordings are needed for those measurements. Follow [EVALUATION_SPEC.md](EVALUATION_SPEC.md) and [LAB_TEST_PROTOCOL.md](LAB_TEST_PROTOCOL.md).
