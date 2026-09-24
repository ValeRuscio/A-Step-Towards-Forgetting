# Validation

Completed locally on 14 September 2026 using Python 3.9, PyTorch 2.8.0, Transformers 4.57.6 and CPU.

- Self-tests passed: exact H0/H1 persistence for a square; centered-CKA and distance invariance under orthogonal rotation/translation; covariance normalization; RoPE identity; common-position-shift invariance; shared Q/K rotation invariance; unique counting of tied matrices.
- Complete tiny-model run: random control, initial tiny model (the smoke fixture's pretrained-labelled slot), A anchor, B-after-9, and both paired comparisons. Six jobs completed.
- Every unique matrix was measured: 16 per tiny snapshot. This tiny configuration is not the real pretrained 1B model.
- Maximum attention reconstruction discrepancy across 32 head/prompt records: 5.2696359e-08.
- Maximum relative local matrix-change reconstruction discrepancy across 60 Linear-use comparisons: 1.7959461e-05.
- Local HF checkpoint loading and association .pt loading were tested.
- Optional unlabelled text geometry was tested; placeholder target loss was not reported as a measurement.
- Notebook schema validated and every code cell compiled.
- Detached launch, duplicate-launch refusal, manual status, completed-job resume and changed-design refusal were tested.
- Both plot types were visually inspected. Included example images are explicitly tiny-model implementation outputs.

No full OLMo-1B production run was performed here. GPU runtime/memory and compatibility with the original Transformers 4.46.3 environment remain to be verified on your machine. Older eager-attention reconstruction compatibility code was not exercised with 4.46.3 locally. Source replay, matrix residuals and attention reconstruction audits must be inspected in the production outputs.

NumPy emitted floating-point warnings in this macOS test environment; resulting audited arrays and all JSON outputs were finite, and the stated numerical checks passed. Nonfinite topology input is rejected and JSON disallows nonfinite values.

These tests establish implementation behavior on the tested configuration, not a cause of forgetting or confirmation of the paper's reference-frame interpretation.


Launcher fix validation (2026-09-14): verified automatic new-folder selection for changed settings and code; duplicate launch retains the same PID; immediate restart stops the prior launcher and launches in a fresh folder; full six-job tiny CPU model run completed; explicit stop ends the launcher; notebook schema and code cells validated. Existing result settings were verified unchanged. Process enumeration is restricted in the local macOS sandbox, so shutdown also handles that case using the verified detached process group. These checks do not establish full OLMo GPU runtime compatibility.


Complete notebook package: executed the actual Start cell with tiny CPU checkpoint configuration, then the actual Refresh cell. All six jobs completed, compact numeric report printed, both saved plots displayed. Verified repeated start keeps the same run, persisted output discovery works without passing SOURCE, and stop on a completed run is harmless. Notebook schema and all code cells validated. Full OLMo GPU execution still requires testing on the user's machine.
