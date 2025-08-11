"""
Pytest configuration file for OAM 6G project.

This file helps pytest discover and import the package correctly.
"""

import os
import sys
from pathlib import Path

                                         
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
