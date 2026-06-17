from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e11_run_cifar100_resnet_lt_tuned_benchmark import all_settings, write_settings_registry


def main() -> None:
    path = write_settings_registry(all_settings())
    print(f"saved tuned benchmark settings registry to {path}")


if __name__ == "__main__":
    main()
