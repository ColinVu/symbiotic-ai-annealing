See https://github.com/Evan-H-Rosenthal/Symbai-Lab-Setup for more details:

To run:

  python3 -m symbiote_weak_generalized.cli.main train \           
  --videos ./hmm-testing/picklist_videos \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --compact-frame-indexing opencv0 \
  --output-dir ../models/classifier \
  --frame-skip 5 \
  --ilr-epochs 1000

  picklist videos must be downloaded from https://gtvault-my.sharepoint.com/:f:/g/personal/fadekola6_gatech_edu/IgA8ZqkeY111Srcl8D94y5l2AUGtzKAmCYXuOoD7GM-pNkk?e=1XBiUU

  NOTE: this current repo version is functional for Shelf C. Instrumentation will be added shortly to add compatibility with the full dataset.
