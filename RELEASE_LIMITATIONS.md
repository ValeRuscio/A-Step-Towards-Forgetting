# Provenance and release boundaries

## What this release establishes

Source files copied from distributed suite archives and nine result exports are preserved byte-for-byte, with provenance hashes. Notebooks had no saved execution outputs at assembly. Authoring code and notebooks were parsed, but the long GPU studies were not rerun during packaging.

## What is not established

- A distributed suite is not automatically the exact version executed remotely. Recorded-run snapshots are kept separately; use their hashes. Remote ad-hoc changes absent from exports cannot be recovered from this collection.
- The seven-case history export is a frozen subset of the twelve-case source study, not a completed balanced twelve-case replication.
- The new Qwen text/history run was still running. The code is included; there is no final exported executed-code/settings snapshot in this collection. Partial results pasted in conversation are not silently converted into a final run manifest.
- The natural-components output-head repair is included in the fixed distributed suite. The earlier broken ZIP is intentionally not offered as the recommended runnable version. Repair scripts are also retained separately.
- `recorded_runs/history_geometry_loose/settings.json` was supplied as a standalone attachment. Its association with the matched-history follow-up must be verified against that remote run before using it as an authoritative configuration.
- Full source SQLite databases, tokenized data populations, remote checkpoints and historical environments are not bundled. Some replay suites require them. Compact calibration/development records embedded in original suites are included because their frozen predictors depend on them.
- OPEN-1B pilot and optional optimizer code are included for completeness, not as claims of completed experiments. Do not infer results from code availability.
- Exact bitwise replay is not promised across GPUs, kernels or package versions. A failed hash check must not be bypassed to make a result appear reproduced.

## Before final submission

1. Export each remaining remote run with its executed code, settings, environment, code hashes and numerical audit summaries; retain the exported file hash.
2. Add an explicit completed/partial/failed status for every cited run and freeze manuscript denominators at that export.
3. Supply required source data records in a separate data supplement, or a documented retrieval route, subject to dataset licenses.
4. Verify that each final table/figure points to the correct recorded run, population and analysis version.
5. Check supplementary anonymity: unchanged historical notebooks/settings may contain original local paths and host metadata. This collection preserves provenance rather than silently redacting it. A blind-review copy should redact identifying metadata consistently and publish a separate manifest of those transformations.
6. Choose an appropriate source-code license; no ownership or third-party license is invented by this packaging step.
