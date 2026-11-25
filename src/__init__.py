"""Makes ``python -m src.*`` work from the repository root.

The env guard is set before any submodule imports scikit-learn, because joblib
tries to detect physical CPU cores with ``wmic`` on Windows, which newer Windows
builds no longer ship. Without it every run prints a multi-line traceback.
"""

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
