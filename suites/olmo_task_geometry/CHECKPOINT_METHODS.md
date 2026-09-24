# OLMo structure study: PH, RMT, reference frames and empirical NTK

This is the original experiment package extended in place. It contains the same two Python files and the notebook; there is no additional runner or wrapper module. All analysis functions are in `structure_study.py`; the original checkpoint/data/replay implementation is included in `forgetting_mechanisms.py`.

## Run

1. Extract the ZIP into a folder on your GPU machine, keeping both Python files beside `Run_structure_study.ipynb`.
2. Open the notebook with your existing OLMo environment. It discovers the original association campaign, including the known `/home/ubuntu/1/runs/olmo_association_v1` location. If your source is elsewhere, set SOURCE in the notebook. No model checkpoints are included in the ZIP.
3. Run the notebook from the top. The Run / restart cell stops the current run tracked by SETTINGS, waits for shutdown, and starts fresh in a separate folder. Previous results are preserved. Rerun that cell to restart; use the existing STOP control to stop without restarting.
4. Rerun Refresh status or Show results manually. There is no continuous notebook output. Show results prints compact numbers and displays the saved plots. Full tables are optional. The diagnostics cell includes an optional EXPORT switch to ZIP all results.

The original exact association replay, all-weight matrix analysis, full A/B risk evaluation, attention audits, and random controls remain. No new training regimen is silently substituted. A fresh random model is a control, not the actual initial checkpoint of pretraining.

## What was added

- Joint persistent homology and covariance spectra on the **same fixed activation point clouds**, instead of comparing unrelated summaries.
- Gaussian, shuffled-entry, and spectrum-preserving point-mixing controls. Each control is feature-centered before analysis. Spectrum-preserving point mixing keeps the centered Gram eigenvalues but can change individual pairwise distances and PH. It is different from a feature-space orthogonal rotation, which preserves all distances.
- Exact finite H0/H1 bottleneck matching, including matches to the diagonal. Activation checkpoint comparisons use the reference checkpoint's distance scale for both clouds, retaining changes in scale. Within-snapshot controls use their observed cloud's scale. Infinite H0 bars are omitted from finite matching; equal-cardinality complete Euclidean filtrations have the same single essential H0 class.
- Attention PH comparisons for both Hellinger distances between attention rows and the nonmetric `1 - symmetrized_attention` flag filtration from the original code.
- A common orthogonal alignment fitted on the first half of prompts and evaluated on the second half. Identity/raw and scale-only differences are reported alongside the fit. This is a **geometric diagnostic**, not a causal activation-patching or loss-rescue experiment. No labels are fitted.
- Empirical task-coordinate NTK over **all trainable parameters**, with per-layer contributions and approximation audits.
- Joins between PH, RMT, weight changes and layer NTK changes. Correlations are descriptive; layers, heads and prompt sites are not independent experimental replicates.
- Direct consecutive B-checkpoint comparisons, including step 21 → step 47 by default.
- CKA returns null with fewer than three observations, avoiding the misleading two-point score of 1.

## Defaults and cost

Geometry uses eight prompts per A/B domain, eight sampled positions per prompt, and sixteen fixed topology landmarks. PH is H0/H1 only, with a hard maximum of 32 landmarks. Absence of an H1 loop is not proof that representations have collapsed. Larger panels and landmark sensitivity checks are needed before drawing a scientific conclusion.

Joint nulls use eight repetitions per activation site. Their ranges are descriptive, not calibrated p-values. RMT comparisons are on fixed samples; the MP edge is only an asymptotic iid reference. Centered, correlated activations do not satisfy the iid assumptions. Do not interpret crossing the MP edge as a formal significance test.

NTK uses four fixed prompts per domain and two scalar coordinates per prompt, giving a 16-by-16 kernel. Each prompt requires two backward passes. The OLMo default projects each parameter-group gradient to 512 coordinates using two independent CountSketch maps, while accumulating exact diagonal gradient norms. Sketch maps are shared across prompts and checkpoints, and groups are summed without cross-group products. Increase `ntk_sketch_dim` if the projection audit fails or the effect is small relative to approximation uncertainty.

The projection reduces stored Jacobian size, **not backward-pass memory**: an OLMo backward pass still requires the model, activations, and a full gradient tuple. Gradient storage alone is approximately four bytes per trainable parameter. All-parameter OLMo runtime and GPU memory have not been measured locally. The notebook runs on the user's existing GPU environment; it does not reinstall packages automatically.

`ntk_mode="auto"` computes an exact kernel only if the Jacobian has at most `exact_max_elements` entries (two million by default); otherwise it uses sketches. Explicit exact mode refuses oversized allocations. Per-group kernels are always sketched, even when the total kernel is exact. Two-sketch disagreement is an audit, not a rigorous confidence interval. The audit prints clearly when it fails.

## What the NTK means

For each prompt the output map is

- answer log-odds: `m = z_red - z_blue`;
- support log-odds: `t = logsumexp(z_red,z_blue) - logsumexp(other logits)`.

The association KL is exactly

`q0 * softplus(-m) + q1 * softplus(m) + softplus(-t) - H(q)`.

Thus its derivative in these coordinates is `(sigmoid(m)-q0, sigmoid(t)-1)`. The kernel is `J J.T` for this finite-network, two-coordinate output map. It captures derivatives sufficient for this task loss. It is **not** the enormous full-vocabulary output NTK, nor an infinite-width theorem, Fisher matrix, or Hessian.

The code combines the kernel with task loss derivatives to report loss-gradient cosine and the first-order old-loss change under infinitesimal B **SGD**. These are not predictions of the historical Adam update: momentum, preconditioning, finite step size and curvature are not included. The kernel panel uses `panel_seed` for all checkpoints so it stays comparable; full evaluation loss still covers all requested seeds. Additional seeds require independent panel runs for replication.

## Where the numbers are

Each snapshot folder contains the original files plus:

- `ph_rmt_joint.json` / `.csv`: diagrams, spectra, null measurements and shared-cloud summaries.
- `ntk.json`: prompt identities, coordinates, exact/sketched mode, audits, loss-gradient statistics and layer summaries.
- `ntk_arrays.npz`: total kernel, independent sketch kernels, sketch features and exact diagonal values.

Each `compare_*` folder contains the original comparison files plus:

- `ph_rmt_frame_changes.json` / `.csv`: bottleneck distances, spectral changes, held-out alignment.
- `attention_ph_changes.json`: per-head comparisons for both attention filtrations.
- `ntk_changes.json`: kernel differences and audits at both endpoints.
- `ph_rmt_ntk_layer_join.json`: per-layer NTK change alongside PH and RMT change.
- `geometry_join.json`: matrix/activation join and a descriptive correlation.

At the results root: original reports/plots plus `extended_report.txt`, `ntk_overview.png`, `ph_rmt_frame_overview.png`, and `ph_rmt_example.png`. The example persistence diagrams and spectrum use the same declared layer-0 A-query cloud; an empty H1 diagram is a valid outcome.

## Interpretation

PH cannot detect a rigid rotation by itself. A stable spectrum plus stable PH is compatible with a rotation, but does not prove one. Held-out frame alignment tests whether one orthogonal map explains drift; with limited calibration data it is underdetermined outside the calibration span, where this implementation acts as identity. Scale-only changes are also tested. The present extension does not perform an inverse-rotation causal rescue.

The spectrum-preserving point-mixing control asks whether observed topology contains structure beyond the covariance eigenvalues. NTK asks how task output errors couple through parameter sensitivities. Agreement between these diagnostics is an association until a targeted intervention is run.

References:
- Ruscio, Nanni, Silvestri, *What are you sinking? A geometric approach on attention sink*: https://proceedings.neurips.cc/paper_files/paper/2025/hash/b2c4b7d34b3d96b9dc12f7bce424b7ae-Abstract-Conference.html
- Doan et al., *A Theoretical Analysis of Catastrophic Forgetting through the NTK Overlap Matrix*: https://proceedings.mlr.press/v130/doan21a.html
