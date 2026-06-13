# Frame Similarity Visualizer - Expected Output Example

## What Each PNG Will Show

### Layout Description

Each generated PNG contains:

1. **Query Frame Section** (Top)
   - Large thumbnail (300x300px) of the query frame (hand-cropped)
   - Item label (e.g., "c11", "c25")
   - Source information (video stem, segment index, frame number)

2. **Comparison Grid** (Bottom)
   - Up to 20 comparison frames in a 4-column grid
   - Each frame is 150x150px
   - Sorted by cosine similarity (highest first)
   - Color-coded borders:
     - **GREEN border** = Same item as query
     - **RED border** = Different item from query
   - Each comparison shows:
     - Item label
     - Cosine similarity score
     - Source information

### Visual Mockup

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ┌────────────────────────────┐                         ║
║   │                            │                         ║
║   │    Query Frame Image       │                         ║
║   │    (300 x 300 px)          │                         ║
║   │    Hand-cropped item       │                         ║
║   │                            │                         ║
║   └────────────────────────────┘                         ║
║   Query: c11                                              ║
║   picklist_061#seg2 frame1234                            ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   ┏━━━━━━━━┓ ┏━━━━━━━━┓ ┌────────┐ ┏━━━━━━━━┓          ║
║   ┃ Frame1 ┃ ┃ Frame2 ┃ │ Frame3 │ ┃ Frame4 ┃          ║
║   ┃150x150 ┃ ┃150x150 ┃ │150x150 │ ┃150x150 ┃          ║
║   ┃ GREEN  ┃ ┃ GREEN  ┃ │  RED   │ ┃ GREEN  ┃          ║
║   ┗━━━━━━━━┛ ┗━━━━━━━━┛ └────────┘ ┗━━━━━━━━┛          ║
║   c11         c11         c25        c11                 ║
║   cos=0.856   cos=0.743   cos=0.234  cos=0.698          ║
║   pick_071#s1 pick_091#s0 pick_081#s2 pick_111#s3       ║
║                                                           ║
║   ┏━━━━━━━━┓ ┌────────┐ ┏━━━━━━━━┓ ┌────────┐          ║
║   ┃ Frame5 ┃ │ Frame6 │ ┃ Frame7 ┃ │ Frame8 │          ║
║   ┃150x150 ┃ │150x150 │ ┃150x150 ┃ │150x150 │          ║
║   ┃ GREEN  ┃ │  RED   │ ┃ GREEN  ┃ │  RED   │          ║
║   ┗━━━━━━━━┛ └────────┘ ┗━━━━━━━━┛ └────────┘          ║
║   c11         c13        c11        c21                  ║
║   cos=0.645   cos=0.189  cos=0.601  cos=0.156           ║
║   pick_121#s4 pick_041#s1 pick_071#s5 pick_051#s0       ║
║                                                           ║
║   ... (more rows of 4 frames each, up to 5 rows)        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Legend:
  ┏━━━┓ Green border = Same item as query (c11 vs c11)
  ┌───┐ Red border   = Different item (c11 vs c25, c13, c21, etc.)
```

## Interpretation Guide

### High Similarity (cos > 0.7)
- Indicates strong feature alignment
- Typically appears for same-item comparisons (but not always!)
- If same-item pairs have low scores, suggests high within-item variance

### Medium Similarity (0.4 < cos < 0.7)
- Moderate feature overlap
- Could be same item with different pose/lighting
- Could be different items with similar visual features

### Low Similarity (cos < 0.4)
- Little feature alignment
- Expected for different items
- Concerning if same-item pairs score this low

### Visual Analysis Questions

When examining the PNGs, ask:

1. **Same-item consistency**: Do green-bordered frames cluster at the top (high similarity)?
2. **Item discrimination**: Are red-bordered frames clearly at lower similarity?
3. **Outliers**: Are there green frames with low scores? (within-item variance)
4. **Confusion cases**: Are there red frames with high scores? (cross-item similarity)

## Example Insights

### Good Embedding Quality
```
Query: c11
  Top 5 comparisons:
    ✓ c11 (0.89) - same item, high similarity
    ✓ c11 (0.85) - same item, high similarity
    ✓ c11 (0.82) - same item, high similarity
    ✗ c25 (0.34) - different item, low similarity
    ✗ c21 (0.31) - different item, low similarity
```
→ Strong same-item clustering, good item discrimination

### Poor Embedding Quality
```
Query: c11
  Top 5 comparisons:
    ✓ c11 (0.45) - same item, but low similarity!
    ✗ c13 (0.43) - different item, similar to same-item
    ✓ c11 (0.39) - same item, even lower
    ✗ c25 (0.38) - different item, similar to same-item
    ✓ c11 (0.36) - same item, very low
```
→ High within-item variance, poor item discrimination

### Confusable Items
```
Query: c11
  Top 5 comparisons:
    ✓ c11 (0.88) - expected
    ✗ c13 (0.85) - HIGH similarity to different item!
    ✓ c11 (0.81) - expected
    ✗ c13 (0.79) - another c13 with high similarity
    ✓ c11 (0.76) - expected
```
→ Items c11 and c13 may be visually similar (check if they're actually different items)

## File Naming Convention

```
similarity_{index:03d}_{video_stem}_seg{segment_idx}_{item_label}.png
```

Examples:
- `similarity_000_picklist_061_seg0_c25.png` - First sample, from picklist_061 segment 0, item c25
- `similarity_019_picklist_121_seg4_c11.png` - 20th sample, from picklist_121 segment 4, item c11

## Summary JSON

Alongside the PNGs, `summary.json` contains:

```json
{
  "total_frames_extracted": 456,
  "unique_items": 18,
  "n_visualizations": 20,
  "hand_neutralize_components": 50,
  "comparisons_per_query": 20,
  "frames_per_item": {
    "c11": 45,
    "c12": 38,
    "c13": 29,
    ...
  }
}
```

This helps you understand:
- Dataset coverage
- Class balance
- How many frames per item are available for comparison

## Tips for Visual Inspection

1. **Look for patterns**: Do certain items consistently show low same-item similarity?
2. **Check frame quality**: Are low-similarity same-item pairs due to blur, occlusion, or lighting?
3. **Compare with/without neutralization**: Run with `--hand-neutralize 0` and `--hand-neutralize 50` to see the effect
4. **Examine confusion cases**: High cross-item similarities reveal which items are hard to discriminate
5. **Verify labels**: If same-item frames look very different, double-check ground truth labels

## Next Steps After Generating PNGs

1. **Quick scan**: Look at 5-10 PNGs to get a feel for typical patterns
2. **Identify problematic items**: Which items have low same-item similarity?
3. **Identify confusable pairs**: Which different items have high cross-similarity?
4. **Compare with heatmaps**: Cross-reference with `embedding_analysis` output
5. **Decide on improvements**: Based on visual patterns, choose preprocessing strategies (from my earlier suggestions)
