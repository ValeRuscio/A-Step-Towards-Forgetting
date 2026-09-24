# Validation

Tested locally with Python 3.9, PyTorch 2.8, Transformers 4.57.6, NumPy and CPU. The real GH200/OLMo-1B campaign was not run locally; its completion rate, GPU peak memory and total disk footprint are not known in advance.

## Completed end-to-end test

A tiny original-OLMo source campaign with four entities, two layers and two heads ran all nine scheduled job kinds: path, geometry, factorials, continuations, gradient/momentum diagnostics, matrices, whole-layer activation patches, NTK and individual-head patches. All completed successfully. The full source training model and optimizer were restored from its saved fork.

The numerical verification checked:

- Every saved scientific JSON numeric value is finite.
- Every path loss panel contains all four entities for each of the four task/prompt splits.
- Per-parameter contributions sum to the reported directional slopes.
- Independent original Adam replay matches the measured path endpoint and the unmodified two-step continuation, at absolute/relative tolerances of 1e-7 or tighter.
- The fully-later factorial corner matches the common unpatched endpoint.
- All whole-layer/head sham patches pass their loss audit. Maximum tiny-test sham loss error: approximately **1.95e-8**.
- The routing/value finite-change identity reconstructs the attention-output change. Maximum reconstruction norm error: approximately **6.84e-17**.
- An independently constructed full tiny-model task-coordinate Jacobian matches the stored exact NTK diagonal within 1e-5. The two-sketch kernel has relative Frobenius error approximately **0.0307** versus the exact tiny kernel.
- Notebook schema validates; all cells compile and have no saved output.
- Both runtime Python modules compile.
- The automatic ZIP passes its integrity check, contains terminal scientific status, and excludes raw arrays when configured to do so.
- Scientific loss/slope, continuation and intervention-heatmap layouts were visually inspected.

## Process/budget test

A separate detached test used a deliberately short session budget. It verified:

- duplicate launch retains the active PID;
- deadline expiry stops the session with `budget_paused`;
- a results ZIP is generated on budget expiry;
- resume starts a new session budget in the same compatible directory;
- previously completed job markers are unchanged and further work progresses;
- restart preserves the previous directory and chooses a new one;
- stop terminates the tracked background process group.

The retained mathematical self-test passes its PH square, rotation invariance, covariance normalization, RoPE and tied-weight tests.

## Reproduce a small local execution

With dependencies installed, from the extracted package:

```
python structure_study.py self-test
python structure_study.py smoke --output /a/new/empty/smoke-folder
```

The smoke command generates synthetic tiny-model checkpoints and runs the overnight pipeline with reduced sizes. Its output is an implementation test, not a scientific forgetting result. Use a new folder each time unless intentionally testing resume.

The local macOS library environment emits an OpenSSL compatibility warning and can emit NumPy/BLAS matmul warnings; numerical outputs and independent checks were verified. No global warning suppression was introduced into the experiment.

Passing a tiny test does not establish the scientific hypothesis. Sham and reconstruction audits remain mandatory in real results. Kernel projections are approximate; topology and RMT are descriptive; activation interventions can create inconsistent intermediate states; retrospective event/head choices are not independent validation.
