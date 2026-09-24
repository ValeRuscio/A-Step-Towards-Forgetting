# OLMo structure study

A full Python implementation and a quiet notebook wrapper for comparing fresh random initialization, a pretrained OLMo checkpoint, the A-learning anchor, and B fine-tuning checkpoints. The comparison is observational: it connects structural change to measured forgetting, but does not by itself establish causation.

## Run

Keep these files together:

- `Run_structure_study.ipynb` — configure, launch, manually refresh status, then show numbers and plots.
- `structure_study.py` — all new weight, activation, attention, topology and comparison functions.
- `forgetting_mechanisms.py` — bundled original OLMo/data/checkpoint core. No dependency on another directory.

Use your existing OLMo/PyTorch environment. The notebook finds the original association campaign near the working directory or under `~/1/runs`. It requires that campaign's **config, data hashes, A anchors and saved forks**, not the measurement ZIP. In your prior environment this is `/home/ubuntu/1/runs/olmo_association_v1`.

Defaults: two fresh random initialization seeds, the source's pretrained checkpoint, seed 1's A anchor, and states **after** B updates 21 and 47. The last two are replayed using the original minibatches and optimizer state. The original source is read only. Change `seeds=[1,2,3]` to expand the campaign. The launcher automatically selects a fresh output directory when settings, code, or library versions change. It updates `SETTINGS["output"]`; use that value for status and results.

The Launch cell returns immediately. Refresh status is manual. Results are written to disk even if you do not display them. Completed jobs are skipped on identical-design resume; unfinished jobs restart. Stop is optional and disabled so Run All will not cancel the new job. Linux/macOS process management is supported; the production target is your Linux GPU machine. Kernel restart ordinarily does not kill the detached process, but server/container/machine termination can.

Do not edit or replace files while a campaign is running. Full-scale runtime has not been measured here. Each original OLMo matrix receives a leading-SVD computation, so start with the default single training seed for timing. Snapshot workers hold one full model; comparison workers hold two full models on CPU and stream matrices to GPU. Loading a source checkpoint can transiently load its optimizer tensors too. Plan for substantial host RAM and several GB of disk for sampled arrays. This code does not form a full Hessian, full Fisher matrix, or a model-sized dense covariance matrix. Keep the existing GPU environment that successfully ran your association experiments.

## What the random comparison does and does not mean

`random_101` and `random_202` are fresh models initialized by the same Transformers OLMo architecture/configuration. They retain architecture, parameter shapes, parameter tying and positional encoding. They are **not OLMo's actual historical initialization**. The initialization distribution is that of the installed Transformers implementation, which need not be identical to the native OLMo pretraining implementation.

Comparing those random states with trained checkpoints shows structural differences associated with training. It cannot tell when the structures emerged, or what the actual pretraining update direction was. Add real intermediate pretraining checkpoints if you have them. Random-to-pretrained parameter subtraction is deliberately not treated as a learning trajectory.

Each matrix also receives two empirical reference distributions on its fixed sampled submatrix: an iid Gaussian and an entry-shuffled copy. The latter preserves the sampled scalar weight distribution while destroying most spatial organization. These are finite null references, not calibrated p-values. Two random models are descriptive controls, not a large replication study; more seeds can be added to `base_checkpoints`.

## Every unique weight matrix

All trainable parameters with two dimensions are enumerated, including embeddings, attention Q/K/V/O, MLP gate/up/down and output weights. Shared parameters are counted once and all aliases are recorded. Each use of a shared weight can still have its own activation analysis. Non-matrix parameters are inventoried separately rather than silently called matrices.

For every matrix:

- Exact Frobenius norm, scalar mean and standard deviation.
- Randomized leading singular values and leading energy fraction, with left/right singular-triplet residual audit.
- An estimated stable rank `||W||_F^2 / estimated_sigma_1^2`. Because the leading singular value is approximate, this is an estimate, not an exact rank or capacity measurement.
- An exact covariance spectrum on a **fixed row/column subsample**, scalar-centered and variance-normalized.
- Marchenko–Pastur asymptotic support and empirical spectral distance to Gaussian/entry-shuffled reference samples.
- H0 and H1 persistent homology of fixed sampled **rows in their full input-coordinate space**.

Leading-SVD analysis uses the full matrix, but does not estimate its entire singular spectrum. Effective rank is reported for the exact sampled covariance spectrum, not extrapolated to the full matrix. Sampling indices and seeds are shared across checkpoints with matching matrix names/shapes.

The default covariance sample is at most 128 by 128, so its aspect ratio often equals one even for a rectangular full matrix. The reported Marchenko–Pastur aspect ratio belongs to that sample. The theoretical support is not applied to raw directed attention eigenvalues. Outliers above the asymptotic edge are descriptive and can occur under a finite random null.

## Paired fine-tuning comparisons

By default, compare A anchor against pretrained, and each B state against its own A anchor. These are same-lineage weight differences.

For every paired matrix:

- Relative displacement norm and the signed cosine between the update and reference weight matrix.
- Leading update singular values with residual audit.
- Principal angles between leading left and right singular subspaces.
- Update energy inside the reference left/output and right/input singular subspaces.
- Fixed-submatrix entry correlation.

Principal subspaces can be unstable near spectral degeneracies or an ambiguous truncation boundary; inspect saved singular values and residuals before interpreting angle changes. These are angles between linear subspaces, not a single physical rotation of an entire network.

For every Linear use of that matrix, sampled inputs and outputs give the finite identity

`delta_y = delta_W x0 + W0 delta_x + delta_W delta_x`.

The code records direct-weight, changed-input and interaction norms, together with reconstruction error and centered CKA. Original OLMo projections are bias-free; a bias-bearing module is explicitly refused rather than silently omitting a term. Component norms do not add up to loss contributions and cannot be interpreted as independent percentages of forgetting.

## Activation and attention geometry

By default, use the same first two A-test and first two B-test prompts from a fixed `panel_seed` at every checkpoint. This provides a small, reproducible geometry panel. It does not represent all contexts or all entities. For broader claims, increase the panel deliberately or supply fixed `geometry_texts` in the notebook. Text prompts are not truncated silently. When unlabelled texts are supplied, their artificial two-answer placeholder is not reported as task loss.

Separately, full original A/B test panels are evaluated for each requested training seed. In cross-seed geometry comparisons, the small geometric panel remains fixed while the declared full task risk uses each seed's own learned label assignments. For B-after-t, the scientifically relevant forgetting row is its matching seed's A loss change from that seed's A anchor. Other seed/world evaluations are cross-world probes, not forgetting of tasks that checkpoint was necessarily taught.

Every Linear/Embedding output is sampled at fixed token positions. All attention heads in all layers are inspected on the full selected prompt. We record:

- Attention entropy and first-position mass, excluding the trivial first query when possible.
- The analytic causal-uniform first-position baseline. A triangular causal mask alone already favors early positions; raw first-position attention is not sufficient evidence of a learned sink.
- Incoming attention adjusted by the number of queries that can see a position, Gini inequality, and a reference position. Reference candidates must be visible to at least 30% of queries.
- A 90th-percentile eligible-edge threshold and 40% query-frequency sink definition, with the same minimum visibility criterion.
- Query/key norms, post-RoPE angular alignment with the first/reference key, and relative key norm.
- Real/imaginary components, resultant length and circular mean of amplitude-weighted query/reference-key cross products in the model's half-split RoPE planes. Only queries able to see the reference contribute.
- Reconstructed attention versus actual attention as an audit. Phase/alignment fields are nulled if reconstruction exceeds the declared tolerance; general attention/graph measurements remain available.
- Attention singular-value gap and squared-singular-value participation ratio. These are not eigenvalues of a symmetric covariance matrix.

Phase here is **spatial Q/K orientation in RoPE planes**. It does not measure temporal oscillation or momentum lag. Reference-phase changes are compared only when the selected reference position agrees. A vanishing resultant has no reliable circular mean angle.

Eager attention is used for observation. Original optimizer replay retains the source implementation; old Transformers versions with fixed SDPA module classes are rebuilt into an eager model afterward with exactly the same state dict. This changes the numerical evaluation backend, not the learned weights. Full production replay/evaluation equivalence across library versions has not been established locally; use the source's working environment and inspect the reported audits.

## Topology: explicit mathematical objects

There are three different constructions; they must not be pooled under one unspecified topology:

1. **Weight/activation row clouds:** Euclidean distances between fixed sampled rows. Dividing by each cloud's median positive distance removes overall scale for shape comparison; the raw distances and scale are also saved. This is the topology of a sampled cloud, not the topology of a global learned manifold.
2. **Attention affinity filtration:** symmetrize the directed attention as `(A + A.T)/2`, remove diagonal self-edges, then use `1 - affinity` with zero distance diagonal. This is a symmetric dissimilarity flag filtration and is **not generally a metric**.
3. **Attention-distribution geometry:** Hellinger distance between complete attention rows, `||sqrt(a)-sqrt(b)||/sqrt(2)`, which is a metric. Landmarks are token positions, while each attention distribution retains its complete key support.

H0/H1 persistence is computed by a deterministic boundary-matrix reduction over F2, including edges and triangles. The built-in implementation caps landmarks at 32; default 16 is intentional. It does not require Ripser. The unit square test verifies a cycle born at 1 and dying at sqrt(2). The full filtration is processed; finite intervals, total/max persistence and Betti curves are saved. These are Vietoris–Rips/flag constructions on the declared finite objects, not evidence of holes in the loss landscape.

Undirected attention graphs are thresholded at .001, .005, .01, .02, .05, .1 and .2 after symmetrization. Reported quantities include the combinatorial Laplacian's Fiedler value and Freeman degree centralization. The explicitly defined star-likeness score is hub coverage times one minus leaf–leaf density; this is our stated operational choice, not a claim of exact reproduction of an unspecified implementation in the paper.

## Connection to the supplied paper

Inspired by Ruscio, Nanni and Silvestri, **What are you sinking? A geometric approach on attention sink**, NeurIPS 2025, especially sections 3–4 (attention geometry, graph/topological methods and random-matrix comparisons).

This is not an exact replication. We preserve the distinction between weights, activations, attention probabilities and update dynamics. We do not assume that a simplex constraint proves a unique reference frame or that position zero's identity RoPE rotation by itself causes a sink. Standard RoPE dot products are invariant under a common position offset; a numerical invariance test is included. Model topology and output behavior must be measured, rather than inferred from that coordinate convention.

A simultaneous orthogonal transformation of Q and K preserves their dot products. A geometry change that is merely a coordinate change need not alter attention or imply lost information. The code tests this invariance and centered-CKA/distance invariance; it does not perform a whole-network gauge intervention. The full parameter Fisher/Hessian is not computed, and no rank statistic is labelled information capacity.

## Additional checkpoints

Add entries to `extra_checkpoints` before launch. Examples (replace with real paths):

```python
SETTINGS['extra_checkpoints'] = [
    dict(id='pretrain_step10000', kind='hf',
         path='/actual/local/HF_checkpoint_directory', compare_to='pretrained'),
    dict(id='seed1_A_continuation', kind='checkpoint', seed=1,
         path='/actual/local/checkpoint.pt', compare_to='seed1_A_anchor'),
]
```

HF checkpoints must be original OLMo with compatible architecture/tokenization. Raw native OLMo training formats are not automatically converted; use a HF export or the bundled association checkpoint schema. The `.pt` loader accepts a raw model state dict, `{'model': ...}`, or `{'engine': {'model': ...}}`. Remote HF entries require a pinned 40-character revision. Local model-weight/config files and source checkpoint files are fingerprinted. A final checkpoint plus fresh random initialization is a structural contrast, not a pretraining time series.

The A-continuation example analyzes a checkpoint you supply; this suite does not fabricate or train that control. Optional `geometry_texts=[...]` replaces the geometric probe panel while full association-risk evaluation still runs.

## Files to inspect

At the output root:

- `structure_overview.png`, `weight_change_overview.png`
- `report.txt` and `full_report.txt`
- settings, fingerprints, job list and process/status/log files

Per checkpoint:

- `matrices.json/.csv`, `matrix_arrays/` with sampled data and leading singular vectors
- `activations.json`, `activation_arrays/`
- `attention.json`, `probe_panel.json`, `probe_predictions.json`, `risk.json`
- aliases, non-matrix parameter inventory, checkpoint descriptor, completion marker

Per comparison:

- `weight_changes.json/.csv`, `activation_changes.json/.csv`, `attention_changes.json/.csv`
- `risk_changes.json`: connect structural changes with signed A/B loss change

The notebook prints compact numbers by default, full per-matrix numbers on request. All numerical records are saved regardless. Do not infer a universal cause from structural differences or correlations across a few checkpoints. The next causal test would intervene on a specific identified structure and verify task effects with controls.

## Launcher update

After replacing structure_study.py in an existing notebook session, reload it:

```python
import importlib
import structure_study as study
importlib.reload(study)
study.restart(SETTINGS)
```

`restart` stops the recorded background process group, waits for shutdown, and starts a fresh folder without deleting previous results. `launch` resumes compatible completed work; repeated calls while active report status. Changed settings/code/library versions automatically select a new folder when inactive. If a previous configuration is still running, use `restart` to apply changes. Scientific input fingerprints remain checked by the worker before resuming.

Always refresh with `study.print_status(SETTINGS["output"])` and display with `study.show(SETTINGS["output"])`, since a restart can change the folder.


## Original notebook: run / restart

The original notebook layout and scientific implementation are retained. The Run / restart cell now calls `study.restart(SETTINGS)`: every execution stops the current background study and starts a fresh one, preserving previous results. Refresh and Show results use the new output path automatically. No notebook_runner.py or session wrapper is required. Use the existing optional STOP control to stop without restarting. Running the entire notebook again restarts the experiment.
