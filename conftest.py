"""Pytest bootstrap.

Runs before any test module, so the environment guards apply even though test
files import scikit-learn directly.
"""

import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings(
    "ignore", message="Setting the shape on a NumPy array has been deprecated.*"
)
