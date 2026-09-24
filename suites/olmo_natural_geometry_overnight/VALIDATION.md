# Validation performed locally

- Tiny original OLMo, CPU, torch 2.8.0 / transformers 4.57.6.
- Instrumented versus ordinary Adam training: bitwise identical final weights.
- Native history/current decomposition: actual update reconstruction within tested tolerance.
- Exact mixed Hessian contraction checked against a central finite difference of gradients.
- Pairwise geometry and persistence invariant under a synthetic rotation/translation.
- Completed-run resume produces no extra steps.
- Simulated crash rollback recomputes the same final weights exactly.
- Detached worker start, idempotent start, cooperative stop, resume and changed-settings sibling output all passed.
- Notebook format and all notebook cells/Python modules compile.

The full OLMo-1B workload, GPU memory requirement and 11.5-hour completion fraction were not tested on a GH200 locally. No full-size experiment was launched on your remote server by this delivery.
