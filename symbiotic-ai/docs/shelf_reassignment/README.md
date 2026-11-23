# Shelf reassignment

This component recommends when to move items between shelves to reduce recognition mistakes. It makes a recommendation from observations available at the current simulation step; the simulator decides whether to apply the move. The decision component does not move physical items or change the live classifier.

**Status:** the default implementation now follows the [label-free ambiguity plan](LABEL_FREE_AMBIGUITY_PLAN.md). It works with pickup shelves and embeddings only. `mode="legacy"` keeps the inferred-label path for comparison. Thresholds remain uncalibrated; synthetic checks do not demonstrate accuracy gains.

## Default label-free implementation

For each segment, apply saved transforms and average each shelf candidate's frame cosine distances. Two candidates count as ambiguous when both distances are plausible and their gap is small. Each segment casts at most one unordered-pair vote; at least three distinct observations are required by default. Weighted-vote winners remain available for comparison in segment evidence.

Pair risk combines centroid proximity with repeated observed ambiguity. The optimizer sums risks for pairs sharing shelves and adds a cost per moved SKU. Cross-shelf swaps account for both destination shelves using an incremental risk delta. It retains legal slot assignments, the move cap, optional three-shelf cycles, bounded partner search, and a no-move option. Geometry fallback is limited to shelves with sparse usable history. Shelf-only historical observations are never assigned a guessed SKU or replayed on a proposed shelf.

The output's `metric_type="ambiguity_risk"` distinguishes predicted risk reduction from errors or accuracy. See [integration](INTEGRATION.md) for settings and [synthetic validation](LABEL_FREE_VALIDATION.md) for checks and remaining data requirements.

## Legacy implementation (`mode="legacy"`)

1. **Find confused pairs.** For each observed pick segment with usable image embeddings, the existing shelf-constrained classifier predicts an item. The system compares that prediction with the item's inferred SKU assignment. If a segment assigned to A is predicted as B, it counts one `A → B` confusion. It combines errors in both directions and divides by the total usable segments assigned to A or B. These are decision-time proxy labels, not verified ground truth.
2. **Choose pairs with enough evidence.** A pair can trigger the targeted search when the items share a shelf, at least five usable segments belong to the two items combined, at least two are mistaken for each other, and their combined confusion rate is at least 20%. These starting thresholds are configurable. The search considers up to ten qualifying pairs at a time.
3. **Test ways to separate them.** For each pair, the system considers moving either item to another shelf. Because shelves must remain full, it exchanges that item with an occupant of the destination shelf. It uses observed embedding overlap to rank a limited set of exchange partners. A small fallback search can still propose candidates when confusion evidence is sparse or targeted swaps do not help.
4. **Check the whole layout.** The system replays its weighted recognition vote under each shortlisted layout. A move is recommended only if it reduces estimated segment errors enough to justify the number of items moved, including any new errors on destination shelves. It may recommend no move. After accepting a move within one decision, it updates the confusion pairs before searching again. Three-item cycles remain available when a swap does not improve the result.

The previous all-pairs search remains available as `search_mode="all_pairs"` for comparison. The targeted search is a heuristic: it examines fewer swaps and can miss an improvement outside its shortlist.

## Label-free evaluation limitations

The classifier retains its two strongest candidates and their scores for each observed pick. A pair becomes a move target when both candidates are plausible, their scores are close, and that ambiguity recurs while they share a shelf. Centroid proximity supports the diagnosis, but a move also needs to reduce overlap across the proposed destination shelves. The decision ranks layouts by estimated **ambiguity risk**, not by errors against inferred labels. The existing 5 / 2 / 20% confusion thresholds and proxy-error objective are replaced; the new thresholds require calibration for this new signal.

If a pick is known only to come from a shelf, its old recording cannot be assigned to a particular proposed shelf after a move without guessing which item was picked. Actual error improvement therefore needs separate labeled evaluation or new observations after the move. The [implementation plan](LABEL_FREE_AMBIGUITY_PLAN.md) details the required code, data, and validation changes.

## Output

The **legacy** decision result includes a complete proposed SKU-to-shelf/slot layout, move transactions, estimated proxy errors fixed and newly introduced, and the evidence behind the search. The CLI also writes `confusion_evidence.csv`, which shows pair counts, rates, and reasons a pair did not qualify. The default label-free mode reports candidate-score ambiguity and predicted risk reduction. See the [integration guide](INTEGRATION.md) for the current snapshot format, API, CLI settings, and output fields.

The estimates replay existing embeddings while changing which items compete on each shelf. They do not predict changes in lighting, backgrounds, or camera views after a physical move. Inferred SKU assignments can also be wrong. Verified true labels are used only for separate retrospective analysis; they do not select moves. See the [evaluation specification](EVALUATION_SPEC.md) and [physical lab test protocol](LAB_TEST_PROTOCOL.md) for measuring results.

## Validation status

The focused tests for the **legacy label-dependent implementation** pass. In one [synthetic 120-item comparison](TARGETED_CONFUSION_SEARCH_BENCHMARK.md), targeted search generated 40 candidate swaps versus 5,806 for the all-pairs mode, and both reached zero estimated proxy errors. That result does not validate the label-free method. Complete model and simulation artifacts are still needed to compare search quality on the study data; the checked-in five-fold subset is incomplete.
