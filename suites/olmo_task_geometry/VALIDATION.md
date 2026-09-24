# Validation

Tested locally on a tiny original-OLMo model using CPU, Python 3.9, PyTorch 2.8 and Transformers 4.57.6. Full OLMo-1B GPU runtime and memory have not been verified locally.

Completed checks:

- Detached trajectory run through two consecutive original Adam updates, measuring every source entity in both tasks and both validation/test forms.
- Independent replay without diagnostic probes matches all measured post-update A/B validation/test losses to 1e-7 absolute/relative tolerance. The observations do not change the tested training trajectory.
- Native Adam history/current channels reconstruct the actual displacement to relative error about 3.2e-5 in the tiny test, consistent with floating-point update rounding.
- Symmetric finite differences at both spacings match the exact full A-test gradient dot actual displacement: relative errors below 0.00025 in the tiny test. Per-entity errors are separately recorded in production outputs.
- Final-hidden margin decomposition tested to absolute error below 1e-5.
- Coarsened three-outcome Fisher formula checked against a finite KL expansion.
- Identity-preserving topology tested on a fixed point cloud whose label assignment changes while unlabelled geometry stays unchanged; the opposite-label component merge distances correctly change.
- Label-permutation controls generated with preserved label counts.
- Temporal joins correctly use update t features to annotate update t+1 loss changes, only for consecutive measured updates.
- All JSON numerical measurements checked for finiteness. Missing same/opposite-label neighbor cases are null rather than fabricated distances.
- Notebook schema and all cells compile. Full runtime Python files compile.
- Duplicate launch leaves the current PID unchanged. Compatible resume preserves completed measurement files. Restart creates a fresh directory; stop ends the recorded detached process group.
- Both scientific plot layouts inspected. A panel with no valid opposite-label comparisons is explicitly labelled; the tiny test's A entities happen to share the same target class.

The local macOS NumPy/BLAS environment sometimes emits matmul floating-point warnings. Outputs were finite and the independent numerical checks above passed; warnings were not globally suppressed.

These checks validate implementation and small-model execution. They do not establish a mechanism of catastrophic forgetting. Local hidden sensitivities cover the final position only; the frozen metric coarsens the vocabulary to answer0/answer1/other; finite-difference estimates have numerical limitations; and temporal joins are not a validated predictor. Native Adam channel accounting is not an optimizer intervention.


Checkpoint-input fix: a trajectory-only run with a deliberately nonexistent optional HF checkpoint, unused event and unused panel seed completed successfully. Only its actual source fork was fingerprinted. Verified missing local HF directories produce a local-path error, while active remote repositories still require pinned revisions. Notebook optional-checkpoint setup is gated by the enabled checkpoint-study mode.
