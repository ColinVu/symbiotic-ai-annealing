# processed_features_3d

Generated 3-D feature files consumed by `nn_model`.

Each file is named `picklist_NNN` (no extension). One row per video frame:

```
p_hold direction net_aruco
```

Space-separated floats. Generate them with
`objectdetector/hand_object_detector_100DOH_inference.ipynb`, or copy existing
files here. Contents of this folder are gitignored.
