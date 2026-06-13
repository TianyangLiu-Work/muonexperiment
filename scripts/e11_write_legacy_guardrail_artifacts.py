from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e11_condition_geometry.artifacts import LEGACY_GUARDRAIL_SCRIPTS


def main() -> None:
    for script in LEGACY_GUARDRAIL_SCRIPTS:
        subprocess.run([sys.executable, script], cwd=ROOT, check=True)
    print("saved legacy E11 guardrail artifacts")


if __name__ == "__main__":
    main()
