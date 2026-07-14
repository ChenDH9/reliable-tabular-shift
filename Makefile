
PYTHON ?= python
TASK ?= acsincome_region
ACS_PUMS_ZIP ?=
OUTPUT_ROOT ?= build/one_task

.PHONY: test-package reproduce-artifacts reproduce-one-task reproduce-full clean-build

test-package:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -m package -p no:cacheprovider
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) scripts/verify_release.py --root . --profile package

reproduce-artifacts:
	$(PYTHON) scripts/reproduce_artifacts.py

reproduce-one-task:
	@test "$(TASK)" = "acsincome_region" || (echo "Only TASK=acsincome_region is allowed"; exit 64)
	@test -n "$(ACS_PUMS_ZIP)" || (echo "Set ACS_PUMS_ZIP to the official archive"; exit 64)
	$(PYTHON) scripts/reproduce_one_task.py --acs-zip "$(ACS_PUMS_ZIP)" --output-root "$(OUTPUT_ROOT)"

reproduce-full:
	./reproduce_full_experiment.sh

clean-build:
	rm -rf build
