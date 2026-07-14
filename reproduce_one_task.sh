#!/usr/bin/env bash
set -euo pipefail
: "${ACS_PUMS_ZIP:?Set ACS_PUMS_ZIP}"
exec "${PYTHON:-python}" scripts/reproduce_one_task.py --acs-zip "$ACS_PUMS_ZIP" "$@"
