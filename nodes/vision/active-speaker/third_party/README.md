# Third-party model code

LR-ASD is not on PyPI, so the files the node needs are copied here (not under a `vendor/` folder: .gitignore keeps
those out, for code fetched at run time), unchanged, with its licence. The weights are not here: models.py fetches them
by URL (the same commit) and sha256.

| Directory | Upstream | Commit | Licence | Files |
|---|---|---|---|---|
| `lr_asd/` | https://github.com/Junhua-Liao/LR-ASD | `1b6dcd2d8fc2895683de6508ec6294ec47d388ca` | MIT (`lr_asd/LICENSE`) | `loss.py`, `model/Model.py`, `model/Encoder.py`, `model/Classifier.py`, unchanged |

This is the commit the bake-off ran (tools/perception-bakeoff/asd/vendor/LR-ASD). Its `model/faceDetector` (S3FD) is
not copied: the node scores the faces layer's tracks instead of detecting faces again. models.py imports the files as
the repository's top-level `loss` and `model` modules, as its own scripts do.
