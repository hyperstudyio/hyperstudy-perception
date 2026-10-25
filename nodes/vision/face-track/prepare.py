"""Fetch the face node's weights ahead of a run (`perception setup`, and the driver before a remote run): py-feat
downloads Detectorv2's models (RetinaFace, the multitask model, ArcFace) from hf://py-feat the first time a detector is
built, into the Hugging Face cache the node then reads. Built on the CPU: only the files matter here."""

from feat.detector_v2 import Detectorv2

Detectorv2(device="cpu")
print("py-feat Detectorv2 weights are in the cache")
