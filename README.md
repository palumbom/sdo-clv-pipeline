# <img src="docs/assets/logo.png" height="24"> SDO Center-to-Limb Variability (CLV) Pipeline
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://palumbom.github.io/sdo-clv-pipeline/latest)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://palumbom.github.io/sdo-clv-pipeline/dev/)
[![CI](https://github.com/palumbom/sdo-clv-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/palumbom/sdo-clv-pipeline/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2404.16747-b31b1b.svg)](https://arxiv.org/abs/2404.16747)

This package processes SDO data to recover trends in the center-to-limb variability of the solar radial velocity. v0.x.y is described in [Palumbo et al.
(2024b)](https://arxiv.org/abs/2404.16747). The results of this paper can be reproduced using the [showyourwork workflow](https://github.com/showyourwork/showyourwork) from [this repo](https://github.com/palumbom/sdo-clv).

## Installation

The package is not published to PyPI; install it from this Git repository. The
[`uv` tool](https://docs.astral.sh/uv/) is recommended but not required.

### As a dependency (use the library API)

To import `sdo_clv_pipeline` and call its API from your own code, install it
straight from GitHub:

```bash
uv pip install "git+https://github.com/palumbom/sdo-clv-pipeline.git"
# or, with plain pip:
pip install "git+https://github.com/palumbom/sdo-clv-pipeline.git"
```

Then in Python:

```python
import sdo_clv_pipeline
```

The interactive Qt matplotlib backend is optional; add the `gui` extra if you
want it:

```bash
uv pip install "sdo_clv_pipeline[gui] @ git+https://github.com/palumbom/sdo-clv-pipeline.git"
```

> **Note:** this installs the importable `sdo_clv_pipeline` package only. The
> command-line entry points (`scripts/`, `batch/`) are not packaged — to run the
> full pipeline end-to-end, clone the repo as in the next section.

### For development (or to run the full pipeline)

Clone the repo and let `uv` build the locked environment (this installs the
project in editable mode plus the `dev` dependency group from `uv.lock`):

```bash
git clone https://github.com/palumbom/sdo-clv-pipeline.git
cd sdo-clv-pipeline
uv sync
```

Optional extras are not installed by `uv sync` unless requested:

```bash
uv sync --extra test    # pytest, to run the test suite
uv sync --extra docs    # sphinx + theme, to build the docs
uv sync --extra gui     # PyQt backends for interactive plotting
uv sync --all-extras    # all of the above
```

Requires Python 3.12.

## Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8273623.svg)](https://doi.org/10.5281/zenodo.8273622)
[![arXiv](https://img.shields.io/badge/arXiv-2404.16747-b31b1b.svg)](https://arxiv.org/abs/2404.16747)

If you use this code in your research, please cite the relevant [software release](https://zenodo.org/records/10966689) and [paper](https://arxiv.org/abs/2404.16747).

## Author & Contact
[![GitHub followers](https://img.shields.io/github/followers/palumbom?label=Follow&style=social)](https://github.com/palumbom)

This repo is maintained by [Michael Palumbo](https://michaelpalumbo.me). You may contact him via his email - [mpalumbo@flatironinstitute.org](mailto:mpalumbo@flatironinstitute.org)
