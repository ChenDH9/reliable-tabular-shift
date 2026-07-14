
# Reliable tabular shift: reproducibility release

This directory rebuilds article artifacts and executes one fixed representative model shard for
the archived study. It is a packaging release, not a new analysis; the reported scientific values
come from the archived experimental outputs.

Three levels are available:

- Level A (`make reproduce-artifacts`) rebuilds five figures, four editable tables, the manuscript
  PDF, and the supplement PDF from packaged source data without downloading data or training.
- Level B (`make reproduce-one-task TASK=acsincome_region ACS_PUMS_ZIP=<path-to-csv_pus.zip>`) rebuilds
  the predeclared ACS task data in an isolated runtime directory, executes one representative
  formal shard (split 0 / logistic regression / model seed 0), and compares it with the archived
  reference.
- Level C (`make reproduce-full`) is optional and guarded. It re-executes 54 model shards from
  author-prepared audited Parquet inputs and a complete budget authority. It has not been executed
  during preparation of this release and does not regenerate the downstream article source-data
  files by itself.

Third-party raw data, row-level records, personal identifiers, credentials, historical packages,
Git history, the full intermediate result tree, and the downstream analysis pipeline are not
included. Five of the six task families require manually prepared inputs for Level C. See `QUICKSTART.md`,
`REPRODUCIBILITY_LEVELS.md`, and `data_registry/LICENSE_BOUNDARIES.md`.

## License and citation

Author-created software and software-support files are licensed under the MIT License; see
`LICENSE` and `LICENSE_SCOPE.md`. The MIT License does not relicense third-party dependencies or
datasets and does not cover the manuscript, supplement, figures, tables, protocol records,
archived reference outputs, or article source-data files unless explicitly stated. Data-access and
redistribution boundaries are documented in `data_registry/LICENSE_BOUNDARIES.md`.

To cite this release, see `CITATION.cff`.
