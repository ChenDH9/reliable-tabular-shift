
# Reproducibility levels

## Level A: article artifacts

Inputs are `figure_source_data/` and `manuscript_build/`. No raw data or model fit is used. Expected
runtime is about 2--5 minutes. Output is `build/artifacts/`; the build report records every hash.

## Level B: one representative end-to-end formal shard

The only allowed task is `acsincome_region`; the workflow rebuilds the ACS task data from the
official archive, but executes only the shard with split seed 0, logistic regression,
model seed 0, using formal execution commit `d745e5412c1d530a2ae64e2eaa42c85c1f64e419`. Expected runtime on the audited 40-core server is
5--8 minutes after materialization. Materialization may take several minutes and needs roughly
8 GiB RAM and 5 GiB free disk. Keys, statuses, counts, hashes, missingness and infinities must match
exactly. Finite scientific floats use the predeclared `atol=1e-10`, `rtol=1e-9`. Timestamps,
runtime, commit labels, and portable-config hashes are intentionally excluded.

## Level C: optional 54-shard formal re-execution from prepared inputs

The frozen design contains six tasks and 54 shards. The audited run used eight workers and finished
in 1080.9 seconds wall time; allow 0.5--3 hours depending on hardware and data layout. Recommended
capacity is 16 CPU threads, 64 GiB RAM, and at least 50 GiB free disk. The output directory is
user-selected and shards resume through their completion markers. Level C requires author-prepared,
audited Parquet inputs for all six tasks, a complete 600-row-per-split budget authority, and the
exact formal execution commit. It is guarded by `CONFIRM_OPTIONAL_FULL_REPRODUCTION=YES` and has
not been executed for this release. It does not reconstruct five task families from official raw
downloads, compare a complete downstream results tree with the archive, or regenerate the article
source-data tables by itself.
