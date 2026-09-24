# Validation

Validated locally on CPU with Python 3.9, PyTorch 2.8, Transformers 4.57.6, NumPy, Matplotlib, psutil and nbformat. Full OLMo-1B GPU execution and the lower supported Transformers versions were not tested in this environment.

## Completed checks

- Full detached tiny-OLMo study: five snapshots and four comparisons, nine completed jobs. Includes two consecutive B checkpoints and their direct comparison.
- Original all-matrix, activation, attention and loss outputs plus new joint PH/RMT, orthogonal alignment, NTK, and layer joins generated successfully.
- Exact task-coordinate NTK contracted with analytic loss-coordinate derivatives matches directly differentiated A/B loss-gradient inner products (relative tolerance 2e-5).
- Forced CountSketch path independently compared against the exact tiny-model kernel: relative Frobenius error 0.0006039258 with 4096 coordinates per group and two sketches. This is a validation result for that tiny model, not an accuracy guarantee for OLMo.
- Exact diagonal norms and independent-sketch disagreement saved and audited.
- Finite bottleneck matching tested on identical/empty diagrams, unequal interval counts and diagonal assignments.
- Orthogonal rotations preserve pairwise distances; a known common rotation fitted on calibration points recovers held-out points to numerical precision in a full-span synthetic example.
- Spectrum-preserving point mixing preserves centered Gram eigenvalues to relative error below 1e-12 in a synthetic test.
- Two-observation CKA correctly returns null.
- Every JSON numerical result in the final tiny study was checked for finiteness.
- Both attention filtrations produced comparison records (64 records per comparison in the two-layer tiny test).
- Notebook schema and all code cells validated.
- Duplicate launch keeps the current process. Restart creates a new results directory. Stop terminates the recorded detached group. Previous result files remain intact.
- NTK heatmaps, PH/RMT/frame scatter plots, and example persistence diagrams/spectrum were visually inspected.

The macOS NumPy/BLAS environment emitted floating-point matmul warnings on some operations, as in the prior package. Returned numerical values were finite and the independent algebraic checks above passed; warnings were not globally suppressed.

## Scientific scope

Tests verify implementation identities and successful small-model execution. They do not establish that rotations, topology, anisotropy, or kernel overlap cause forgetting. The joint RMT nulls are descriptive, the PH calculations use finite landmarks, and the frame fit is evaluated geometrically rather than as an intervention on task loss. The kernel's SGD prediction does not model Adam momentum or finite-step curvature.
