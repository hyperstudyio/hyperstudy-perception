EfficientAT's MobileNet (`mn`) model and mel front end, vendored from https://github.com/fschmid56/EfficientAT at
commit a425fdce92572e602a1d5634799bd9f1f2efa806 (MIT, `LICENSE`; Florian Schmid), the commit the bake-off ran. It is
not published on PyPI.

Files: `models/mn/{model,block_types,utils,attention_pooling}.py` as `mn/`, and `models/preprocess.py`. Changes, all to
imports, so the files load as this package next to the node's own `models.py`:
- `from models.mn.X import ...` became `from .X import ...`;
- `model.py` no longer imports `helpers.utils.NAME_TO_WIDTH`, which it did not use, and whose module reads
  `metadata/class_labels_indices.csv` from the working directory when imported.

The weights (`mn40_as_mAP_484.pt`, the release v0.0.1 asset, MIT) are fetched by `../models.py`, checked by sha256;
`get_model` is called without a pretrained name, so this code never downloads anything itself.
