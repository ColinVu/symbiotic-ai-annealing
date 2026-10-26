# Targeted confusion search: initial comparison

## Synthetic comparison

The first comparison used 120 SKUs on five shelves of 24, 120 revealed one-frame observations, 32-dimensional normalized centroids, and a seeded near-duplicate pair on the same shelf. Six observations assigned to one item used the other item's embedding. Both modes used `max_moved=2`, `cycle_pool_size=0`, `swap_replay_limit=50`, and no fallback. Each mode used a fresh `DecisionEngine` on the same snapshot and model. Times below are single wall-clock measurements on the local development host; they are indicative, not a stable performance benchmark.

| Search mode | Time | Generated swaps | Exact swap replays | Final proxy errors | Moved SKUs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `confusion` | 0.047 s | 40 | 40 | 0 | 2 |
| `all_pairs` | 0.194 s | 5,806 | 96 across passes | 0 | 2 |

The targeted mode generated far fewer candidates on this sparse-confusion case and found the same zero-error result. The replay total spans multiple passes, each capped at 50. This is one synthetic case, so it does not establish equivalent search quality across real snapshots.

## Remaining validation

The checked-in `cv_results_subset` does not contain the complete centroid files required for a full simulator comparison. When complete model and snapshot artifacts are available, run both modes with identical settings across early and late simulation steps. Save wall time, candidate/replay counts, proxy objective, retrospective errors where truth is separately available, moved SKUs, and every missed improvement. Use those results to tune the evidence thresholds and decide whether the default quality is acceptable.

The focused shelf-layout suite passes with both modes, including bounded replay, repeated-confusion detection, sparse-evidence fallback, deterministic output, and the existing capacity and regression tests.
