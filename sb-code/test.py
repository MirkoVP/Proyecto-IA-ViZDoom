
from pathlib import Path
import os
CURRENT_DIR = Path(os.path.abspath('')).resolve()
LOG_DIR = CURRENT_DIR.parent / f"trains/take-cover"
LOG_DIR = str(LOG_DIR)
print(LOG_DIR)

