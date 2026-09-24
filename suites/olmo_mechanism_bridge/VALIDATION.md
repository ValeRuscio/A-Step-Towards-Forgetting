# Validation performed

Validated locally on 2026-09-15, with a tiny randomly initialized original-OLMo fixture; no remote model download or full-scale GH200 experiment was used as a test.

Passed:

- Known proper rotation recovered on held-out rows; vector norms preserved.
- Exact routing/content/cross-term algebra.
- H1 persistence invariant under rigid rotation.
- Full four-corner experiment with actual OLMo forward/backward passes, attention reconstruction, fixed-map and random interventions, held-out records, geometry, patched-site and downstream PH.
- Calibration finite-difference checks for local margin projections.
- Equal-norm optimizer direction comparison with identical-origin audit.
- Original, reset-first-moment and quarter-learning-rate continuations.
- Instrumented original continuation matches direct uninstrumented replay on all four loss splits to the tested tolerance of 1e-10.
- Mid-branch checkpoint interruption and resume reproduce the uninterrupted records exactly on the tiny fixture.
- Completed-job resume preserves completion markers.
- Background duplicate launch, stop, resume, changed-settings sibling directory and fresh restart.
- Automatic ZIP export and report/plot generation.
- Notebook schema, all notebook code cells, and Python module syntax.

The final complete smoke campaign finished 5/5 jobs. Plots were visually inspected. The local Python environment emits an unrelated urllib3/LibreSSL compatibility warning; the final smoke run had no failed numerical audits or experiment jobs.

These checks establish tested implementation behavior, not a scientific finding, a guarantee of full-scale runtime, or compatibility with OLMo2/3. Use the existing working GH200 environment and preserve the source campaign's pinned original OLMo revision.
