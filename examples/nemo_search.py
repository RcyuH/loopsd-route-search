"""Run with NeMo RL's Python runtime and this sibling project's source on sys.path.

Example from nemo-rl:
python ../loopsd-route-search/examples/nemo_search.py search \
    --config ../loopsd-route-search/configs/qwen3_1p7b_gsm8k.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loopsd_search.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv.extend(["--backend", "nemo"])
    main()
