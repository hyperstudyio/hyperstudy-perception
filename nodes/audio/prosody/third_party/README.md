# Third-party model code

The Odyssey 2024 speech emotion baseline is not on PyPI: its model card loads its classes with `trust_remote_code`
from the repository's `pipeline_utils.py`, which no longer imports under transformers 5. The classes are copied here
(not under a `vendor/` folder: .gitignore keeps those out, for code fetched at run time), never fetched at run time.
The weights are not here: odyssey.py reads them from the Hugging Face cache at the pinned revision.

| Directory | Upstream | Commit | Licence | Files |
|---|---|---|---|---|
| `odyssey/` | https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes | `00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9` | MIT, per the model card (`odyssey/LICENSE`) | `ser_model.py`, from `pipeline_utils.py`, changed only to load under transformers 5 (its header says how) |

This is the code and revision the bake-off ran (tools/perception-bakeoff/prosody/prosody.py, load_odyssey), whose
check against the card's own code under transformers 4.57.6 on the CPU gave a largest difference of 2.2e-06 over six
OOO utterances.
