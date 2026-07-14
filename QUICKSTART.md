
# Quick start

Python 3.11 is required. Pandoc and Tectonic are required only for Level A PDF builds. A preferred
Conda-compatible setup is:

```bash
micromamba create -n reliable-shift-stage3r -f environment.lock.yml
micromamba activate reliable-shift-stage3r
micromamba install -c conda-forge pandoc=3.10 tectonic=0.16.9 poppler=26.05.0
```

The portable pip fallback is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-export.txt
```

Then run:

```bash
make test-package
make reproduce-artifacts
make reproduce-one-task TASK=acsincome_region ACS_PUMS_ZIP=<path-to-csv_pus.zip>
```

The Level B ZIP must be the official 2018 ACS 1-year person PUMS archive. The script refuses a
wrong size or SHA-256 and never downloads or packages raw data. Default outputs are below `build/`;
existing frozen results are never modified.

The exact document-tool versions are recorded in `DOCUMENT_TOOLCHAIN.lock.md`. The supplied Conda
lock is for `linux-64`; on another platform, create a platform-specific environment that satisfies
the same top-level versions before running the verification commands.
