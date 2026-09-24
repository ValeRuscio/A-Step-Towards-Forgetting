# Local validation

Environment: CPU, torch 2.8.0, transformers 4.57.6; tiny original OLMo, no downloads.

Passed:
- Natural post-update weights identical to an independent ordinary training loop.
- Diagnostic interpolation leaves optimizer states identical.
- Mean per-prompt gradient projection agrees with independently computed full-population gradient.
- Simpson quadrature integrates a known polynomial derivative correctly.
- Path-integrated history/current/roundoff contributions sum to the total derivative integral.
- Saved finite-difference checks and tiny-model endpoint integration closure.
- Balanced prompt/entity/interaction variance decomposition sums to total variance.
- A stop injected after assigning intermediate weights restores natural post-update weights exactly.
- Cached partial path resumes without duplicate records.
- Completed-run resume, report, plots and ZIP export.
- Detached launch, repeated start, cooperative pause, resume and settings-change sibling directories.
- Notebook schema and compilation of all code cells/modules.

No full-size OLMo-1B/GH200 benchmark was run locally. All configured events are retrospective selections. Numerical path-resolution flags remain part of the experimental output; queue completion is not a guarantee of numerical resolution.
