# Third-party model code

Neither project is on PyPI, and torch hub would clone them from GitHub at run time, so the files Gaze-LLE needs are
copied here (not under a `vendor/` folder: .gitignore keeps those out, for code fetched at run time), unchanged unless
noted, with their licences. The weights are not here: models.py fetches them by URL and sha256.

| Directory | Upstream | Commit | Licence | Files |
|---|---|---|---|---|
| `gazelle/` | https://github.com/fkryan/gazelle | `2d78f9f3bd2a5db360354954ec1a4c526ee8ab55` | MIT (`gazelle/LICENSE`) | `gazelle/model.py`, `gazelle/backbone.py` unchanged; `gazelle/utils.py` trimmed to the two tensor helpers `model.py` calls (the rest imports matplotlib and scikit-learn) |
| `dinov2/` | https://github.com/facebookresearch/dinov2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | Apache-2.0 (`dinov2/LICENSE`) | `dinov2/__init__.py`, `dinov2/models/`, `dinov2/layers/` unchanged |

These are the commits the bake-off ran (tools/perception-bakeoff/gaze/vendor). `models.py` builds the ViT-L/14 with
the arguments of `dinov2/hub/backbones.py`'s `_make_dinov2_model` (not vendored: it only adds the torch hub download).
xFormers is not installed (envs/gaze), so DINOv2's attention is the plain PyTorch path, as in the bake-off.
