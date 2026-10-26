# Shelf reassignment

This component recommends when to move items between shelves to reduce recognition mistakes. It makes a recommendation from observations available at the current simulation step; the simulator decides whether to apply the move. The decision component does not move physical items or change the live classifier.

## How it works

1. **Find confused pairs.** For each observed pick segment with usable image embeddings, the existing shelf-constrained classifier predicts an item. The system compares that prediction with the item's inferred SKU assignment. If a segment assigned to A is predicted as B, it counts one `A → B` confusion. It combines errors in both directions and divides by the total usable segments assigned to A or B. These are decision-time proxy labels, not verified ground truth.
2. **Choose pairs with enough evidence.** A pair can trigger the targeted search when the items share a shelf, at least five usable segments belong to the two items combined, at least two are mistaken for each other, and their combined confusion rate is at least 20%. These starting thresholds are configurable. The search considers up to ten qualifying pairs at a time.
3. **Test ways to separate them.** For each pair, the system considers moving either item to another shelf. Because shelves must remain full, it exchanges that item with an occupant of the destination shelf. It uses observed embedding overlap to rank a limited set of exchange partners. A small fallback search can still propose candidates when confusion evidence is sparse or targeted swaps do not help.
4. **Check the whole layout.** The system replays its weighted recognition vote under each shortlisted layout. A move is recommended only if it reduces estimated segment errors enough to justify the number of items moved, including any new errors on destination shelves. It may recommend no move. After accepting a move within one decision, it updates the confusion pairs before searching again. Three-item cycles remain available when a swap does not improve the result.

The previous all-pairs search remains available as `search_mode="all_pairs"` for comparison. The targeted search is a heuristic: it examines fewer swaps and can miss an improvement outside its shortlist.

## Output

The decision result includes a complete proposed SKU-to-shelf/slot layout, move transactions, estimated errors fixed and newly introduced, and the evidence behind the search. The CLI also writes `confusion_evidence.csv`, which shows pair counts, rates, and reasons a pair did not qualify. See the [integration guide](INTEGRATION.md) for the snapshot format, API, CLI settings, and output fields.

The estimates replay existing embeddings while changing which items compete on each shelf. They do not predict changes in lighting, backgrounds, or camera views after a physical move. Inferred SKU assignments can also be wrong. Verified true labels are used only for separate retrospective analysis; they do not select moves. See the [evaluation specification](EVALUATION_SPEC.md) and [physical lab test protocol](LAB_TEST_PROTOCOL.md) for measuring results.

## Validation status

The focused shelf-layout tests pass. In one [synthetic 120-item comparison](TARGETED_CONFUSION_SEARCH_BENCHMARK.md), targeted search generated 40 candidate swaps versus 5,806 for the all-pairs mode, and both reached zero estimated errors. Complete model and simulation artifacts are still needed to compare search quality on the study data. The checked-in five-fold subset is incomplete, so the synthetic result is not a claim about physical or held-out accuracy.
