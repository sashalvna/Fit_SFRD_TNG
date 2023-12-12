"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
import os
from pathlib import Path

os.environ["PATH"] += os.pathsep + str(Path.home() / "bin")

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[0].absolute()

# Absolute path to the `src` folder
src = root

# Absolute path to the `src/data` folder (contains datasets)
data = src / "data"

# Absolute path to the `src/static` folder (contains static images)
static = src / "static"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = src / "scripts"

# Absolute path to the `src/tex` folder (contains the manuscript)
tex = src / "tex"

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = root / "figures"

# Absolute path to the `src/tex/output` folder (contains other user-defined output)
output = root / "output"