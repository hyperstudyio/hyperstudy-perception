"""test.work's prepare (perception setup, a remote run before the node starts): records that it ran, with its settings."""

import json
import sys
from pathlib import Path

config = json.loads(sys.argv[1])
if config.get("record_dir"):
    Path(config["record_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(config["record_dir"]) / "prepared").write_text(json.dumps(config))
    (Path(config["record_dir"]) / "environment").write_text(json.dumps(dict(__import__("os").environ)))
if config.get("fail") == "prepare":
    sys.exit("could not fetch the weights (on purpose)")
print("test.work is prepared")
