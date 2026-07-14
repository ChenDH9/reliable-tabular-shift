#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python}" scripts/verify_release.py --root . --profile package "$@"
