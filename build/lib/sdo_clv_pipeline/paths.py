from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[1].absolute()

# Absolute path to the `src` folder
src = root / "sdo_clv_pipeline"

# Absolute path to the `src/data` folder (contains datasets)
data = root / "data"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = root / "scripts"

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = root / "figures"
