"""models.py without torch: the Apple GPU's macOS minimum, which the manifest's mps runner states as well."""

import importlib.util
import re
from pathlib import Path

# loaded under a name of its own: other nodes have a models.py too, and a test of theirs may have imported `models`
_spec = importlib.util.spec_from_file_location("sounds_models", Path(__file__).parent / "models.py")
models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(models)


def test_the_apple_gpu_is_refused_before_macos_15_1_and_the_manifest_says_the_same():
    assert [models.older_macos(v) for v in ("14.6.1", "15.0", "15.0.1")] == [True] * 3
    assert [models.older_macos(v) for v in ("15.1", "15.1.1", "26.0", "")] == [False] * 4   # "": not a Mac, or not known
    # the mps runner's line (no yaml in the nodes' test environment)
    [mps] = [line for line in (Path(__file__).parent / "manifest.yaml").read_text().splitlines() if "accelerator: mps" in line]
    assert re.search(r'minimum_macos: "([\d.]+)"', mps)[1] == ".".join(map(str, models.MPS_MINIMUM_MACOS))
