# csv_labels

Hand-annotated state labels used to **train** `nn_model`.

Copy files from the segmentation repo (`symbiotic-ai-segmentation/csv_labels/picklist_NNN.csv`) into this folder.

Accepted formats:

- quoted HTK rows: `"0\tm"`
- compact CSV: `frame_index,code` (same as `picklist_labels`)

These are training supervision, not annealing inputs. Bulk NN prediction writes
annealing labels to [`picklist_labels/`](../picklist_labels).
