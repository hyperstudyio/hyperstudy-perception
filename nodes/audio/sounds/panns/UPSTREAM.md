PANNs' Cnn14_DecisionLevelMax, vendored from https://github.com/qiuqiangkong/audioset_tagging_cnn at commit
d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4 (MIT, `LICENSE.MIT`; Qiuqiang Kong), the commit the bake-off ran. It is not
published on PyPI.

Files: `pytorch/pytorch_utils.py` unchanged, and `cnn14.py`, which is `pytorch/models.py` cut down to what
Cnn14_DecisionLevelMax uses: the imports, `init_layer`, `init_bn`, `ConvBlock` and the `Cnn14_DecisionLevelMax` class,
each unchanged, with `from pytorch_utils import` made `from .pytorch_utils import`. The spectrogram and log-mel layers
come from torchlibrosa 0.1.0 (PyPI, MIT).

The weights (`Cnn14_DecisionLevelMax_mAP=0.385.pth`, Zenodo record 3987831, CC-BY-4.0) are fetched by `../models.py`,
checked by sha256.
