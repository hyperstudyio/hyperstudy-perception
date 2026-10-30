"""test.gated's prepare: records what it was given, then puts its weights in the shared weights cache (as
perception_node.weights does), unless told to fail."""

import json
import sys

from record import record, weights

config = json.loads(sys.argv[1])
# and where a stored login would be read from (HF_TOKEN_PATH), and whether one is there
import os   # noqa: E402
record(config["record_dir"], "prepare", token_path=os.environ.get("HF_TOKEN_PATH"),
       stored_token=bool(os.environ.get("HF_TOKEN_PATH")) and os.path.exists(os.environ["HF_TOKEN_PATH"]))
if config.get("fail_prepare"):
    sys.exit("Hugging Face refused this token")
weights().parent.mkdir(parents=True, exist_ok=True)
weights().write_text("test")
print("test-gated-weights is in the cache")
