#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python}" scripts/reproduce_artifacts.py "$@"
