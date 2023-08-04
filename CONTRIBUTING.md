# Contributing to Bandolier

This is mostly notes for myself.

## Package upload

```bash
python3 -m venv venv
. venv/bin/activate
pip install -e ".[dev]"

# bump the version in setup.py
vim setup.py

python setup.py sdist
twine upload dist/*
```
