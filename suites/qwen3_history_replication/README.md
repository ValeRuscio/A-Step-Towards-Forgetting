# Qwen3 history replication — standalone second-GH200 package

Unzip this folder on the NEW Linux GPU server. Open Run_Qwen3_History.ipynb in that folder. No files from the original server are needed.

1. Run the setup cell once. It creates .venv using the notebook's Python (3.10–3.13), installs CUDA PyTorch and dependencies there, and checks a tiny Qwen3 forward, backward and second derivative on the GPU. The existing notebook kernel is not modified or restarted. Installation logs go to setup.log. A driver must already be supplied by the GPU host; verify_gpu.py detects CUDA failure. No manual CUDA toolkit, flash-attention, torchvision or quantization packages are required.
2. Run configuration, then Launch/resume. Training runs in a detached process using .venv/bin/python. You can close the notebook; the GPU server must remain running.
3. Rerun Status to refresh. Reports and plots are displayed only on demand. Stop is cooperative; wait for alive=False, then Launch/resume. Never launch duplicate copies of the package against one output folder.
4. Export creates runs/qwen3_history_v1/qwen3_history_share.zip. Upload that ZIP for analysis. It includes databases, summaries, plots, settings and code, but excludes model checkpoints, cached datasets and prompt text files. Data are reconstructed from pinned datasets and recorded identities.

The notebook defaults to the saved settings when reopening. Scientific settings are immutable within an output directory. To change them, use a new output directory. Do not edit installed code during an active/resumable run. Stop/resume does not need a fresh output directory. Resuming after an interruption may rebuild several inexpensive training updates; committed diagnostic records are reused.

## Registered design

Qwen/Qwen3-0.6B-Base at da87bfb608c14b7cf20ba1ce41287e8de496c0cd, the same revision recorded in your earlier Qwen run. Seeds 61,62,63, each shared/disjoint answer conditions: six trajectories. Fixed FP32 full-parameter Adam, beta1=.9, beta2=.99, eps=1e-8, clip=1, no weight decay. LR=2e-5 carried from the earlier Qwen study; not calibrated against these new confirmation outcomes. This is not a claim that LR is equally optimal across models.

Same text setting as the Pythia/SmolLM history study: SST-2 task A, AG News task B; pinned dataset revisions, hash-separated confirmation content, counterbalanced answer-token codebooks, 600 A updates, 128 B updates, microbatch2 with accumulation2, context256, train1024, valid32, test128. Paired answer conditions retain identical A data/weights and content identities. Acquired flag requires validation loss improvement>=.1 and restricted accuracy>=.6; all runs and acquisition failures remain reported. No automatic retuning or seed substitution.

All full-model operations use FP32/eager attention; reductions use FP64 where the existing implementation did. Flash attention/TF32/AMP are not enabled. This preserves numerical comparability and permits second derivatives. This package does not collect PH, fit a reference frame or train a new predictor: the geometry scores are unfitted, and selection is fixed-rule.

## Automatic stages, in order

1. Natural trajectories for all six cases. Endpoint losses, full/restricted accuracy, answer mass, confusion/leakage, native channels and exact A-valid component gradients. Full test uses128 examples. No path/Hessian work delays collection of the other natural trajectories.
2. Exact source replay for all six cases. Fixed full A-valid32 and first A-test32 gradients; every original A anchor and B endpoint must reproduce exactly. History/current magnitude and task-relative cosine, symmetric cross-time decomposition, age bins, adaptive denominator and clock decomposition, block-level and global records. New age-bin norms/cosines and fixed gradient cohorts are added without changing old projection definitions.
3. Selected finite paths: one uniform update per trajectory (outcome independent), plus up to one draw from A-valid increases>.05. Enrichment uses validation only, not held-out results. Duplicates are combined. At most12 selected updates /24 population paths. Each path uses32 examples; 5->9 Simpson grid points, multi-width finite differences [.04,.02,.01], component Hessian bilinear forms and symmetry/integration checks. Failed/capped audits remain unresolved. Separate FP64 derivative rechecks do not validate native FP32 by substitution. Four fixed prompts get local records. Smaller path coverage/resolution than the original run is explicit; tolerate unresolved outcomes, do not silently loosen thresholds.

A 23.5-hour SESSION budget is not a completion-time estimate. The full study can require multiple sessions, especially path audits. Rerun Launch/resume after a budget pause. Natural/history stages finish before expensive path audits. Do not infer scientific negatives from unfinished work. Stage durations and progress are recorded; timing should be estimated from the first real case rather than the tiny local test.

## Age and cohort semantics

History age bins: task-A terminal moment, B ages1–4,5–16,17+, plus explicit native roundoff residual. New outputs store projections, norms and cosines for each bin, globally and by parameter block. Bin widths, decay and number of available past updates differ; raw signed sums are not per-gradient effects.

Cohort origins are B updates1,8,32,64, fixed before outcomes. Each is followed for ages1–32, on the actual trajectory. At update t, origin k contributes
  u_k(t) = -lr/(1-beta1^clock_t) * D_t * (1-beta1)*beta1^(t-k) * clipped_gradient_k.
It is a contribution to history, not the current channel. A parallel evaluation uses the origin update's denominator D_k. Both use the actual current task gradient, the same current clock and age decay. The frozen-denominator direction therefore remains fixed up to a positive scalar; cosine changes there reflect changes in task-gradient orientation. Current-versus-frozen denominator comparisons isolate the evaluation effect of the chosen scaling at the current state, not a continued-training intervention. No new model branches are trained. Sign flips near zero are descriptive, not significant findings. Tables retain magnitudes and all ages; no post-hoc selected cohort is discarded.

Remainders and mixed Hessian terms are distinct. Total = confusion + leakage; finite loss = first-order projection + finite remainder. Small mixed history/current curvature does not imply history's first-order contribution is small. Cross-time signed sums telescope; use absolute scalar contributions and event windows. Norm/orientation, vector-motion, and numerator/scaling are different (partly nested) decompositions, not additive causal percentages.

## Storage and outputs

Only active model/optimizer checkpoints are retained. No per-prompt full gradient arrays are saved to disk. A rolling16-gradient buffer and a small number of fixed cohorts use host RAM; several tens to over100GiB may be needed with all exact CPU tensors. The GH200's large host memory is useful; its advertised480GB is not all GPU VRAM. Default reserve50GiB; stage checkpoint cap100GiB. Allow additional space for model/dataset cache and export. No cache/research deletion is performed.

Natural outputs: natural_summary.json, event_components.json, answer_condition_pairs.json, path_summary.json and path audits in results.sqlite, REPORT.txt, overview.png.
History outputs: summary.json, replay_audits.json, geometry_summary.json, age_summary.json, global_timeline.json, event_windows.json, age_geometry_timeline.json, cohort_timeline.json, cohort_summary.json, REPORT.txt, overview.png, results.sqlite. Full per-layer cohort and age records are in SQLite; portable JSON timelines are global.

Native terminal momentum from this study is task-trained Adam memory, not original Qwen pretraining optimizer history.

## Setup sources and validation limits

PyTorch provides the CUDA12.8 2.11.0 Linux aarch64 wheel:
https://download.pytorch.org/whl/cu128/torch/
https://pytorch.org/get-started/previous-versions/
Qwen3 requires Transformers>=4.51; this package pins4.57.6, as in your working Qwen environment:
https://huggingface.co/Qwen/Qwen3-0.6B-Base

See VALIDATION.txt. Local tests use tiny models on CPU; the real GH200 package installation and full pretrained run cannot be validated on this Mac. The included setup probe executes on your new GPU before any long job launches.

## ARM64 cuSPARSELt setup check
If pip check reports only `nvidia-cusparselt-cu12 0.7.1 is not supported on this platform`, run repair_gh200_setup.py beside the existing .venv. It accepts only the exact SBSA-tag discrepancy after checking Linux ARM64, package version, WHEEL tag, a64 ELF library and successful dynamic loading, then runs the CUDA/Qwen3 probe. Other dependency errors remain fatal. No vendor metadata is edited and no packages reinstalled. Evidence is saved to dependency_check.json. Updated setup calls this verifier automatically.
