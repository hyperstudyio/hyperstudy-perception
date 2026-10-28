"""Fetch the Odyssey weights ahead of a run (`perception setup`, and the driver before a remote run): its config and
checkpoint, and WavLM-large's config, at the pinned revisions, into the Hugging Face cache the node then reads
(odyssey.py). Nothing runs. Praat comes with Parselmouth, in the environment.
    python prepare.py CONFIG_JSON      (the node's settings, as its request would carry them; it has none)"""

from odyssey import REPO, REVISION, WAVLM, WAVLM_REVISION, files

files()
print(f"{REPO}@{REVISION[:8]} and {WAVLM}@{WAVLM_REVISION[:8]}/config.json are in the cache")
