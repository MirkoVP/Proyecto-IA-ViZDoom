
from pathlib import Path
import os
CURRENT_DIR = Path(os.path.abspath('')).resolve()
REST_DIR = f"trains"
LOG_DIR = CURRENT_DIR / REST_DIR
print(LOG_DIR)

