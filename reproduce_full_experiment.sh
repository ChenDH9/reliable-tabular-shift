
#!/usr/bin/env bash
set -euo pipefail

if [ "${CONFIRM_OPTIONAL_FULL_REPRODUCTION:-}" != "YES" ]; then
  echo "Level C is optional and was not run in Stage 3R."
  echo "Set CONFIRM_OPTIONAL_FULL_REPRODUCTION=YES only after rebuilding and auditing all six datasets."
  exit 64
fi
if [ -z "${FULL_DATA_REGISTRY:-}" ] || [ -z "${FULL_BUDGET_MANIFEST:-}" ]; then
  echo "Set FULL_DATA_REGISTRY and FULL_BUDGET_MANIFEST to audited private inputs."
  exit 64
fi
if [ -z "${FULL_OUTPUT_ROOT:-}" ]; then
  echo "Set FULL_OUTPUT_ROOT to a new, empty output location."
  exit 64
fi
PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}" python scripts/reproduce_full_experiment.py   --data-registry "$FULL_DATA_REGISTRY"   --budget-manifest "$FULL_BUDGET_MANIFEST"   --output-root "$FULL_OUTPUT_ROOT"   --commit "d745e5412c1d530a2ae64e2eaa42c85c1f64e419"
