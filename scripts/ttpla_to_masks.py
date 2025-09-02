#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure local package under src/ is importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wireseghr.data.ttpla_to_masks import main

if __name__ == "__main__":
    main()
