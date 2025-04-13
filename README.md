# Brokest

More broke than a broker.

## Development

### Cloning

First, clone the project

```bash
git clone https://github.com/rentrueawng/brokest
```

### Installation

The dependencies of the package is a little tricky to set up,
so we're using `conda`.

```bash
PYTHON_VERSION_YOU_LIKE=3.12 # I personally like to use newer python.
conda env create -f env.yaml python=${PYTHON_VERSION_YOU_LIKE}
```

Then, install the packages from PyPI, in editable mode.
```bash
# Just the runtime dependencies
pip install -e .

# Development dependencies
pip install -e '.[dev]'
```
