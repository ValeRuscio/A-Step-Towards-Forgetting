# Qwen3 replication and OPEN-1B historical-optimizer pilot

Open `Run_Qwen3_and_OPEN.ipynb`. Keep all Python files together. Run All only constructs controls; it never automatically starts, stops, installs, or downloads models.

## Qwen3: runnable extension of the completed study

- Qwen/Qwen3-0.6B-Base; three seeds, original A acquisition and 128 B updates (384 measured updates for your source configuration).
- Same original task, Adam recipe, fixed plain-text prompts, exact full-population A/B gradients, native history/current channels, recent-vector changes, and original frozen OLMo predictors. No refitting.
- Defaults retain detailed paths on seed 1 (two validation-selected events). Set QWEN['path_seeds']=[1,2,3] before launching to request paths for every seed. This changes cost, not the 384-update trajectory count.
- Forward/gradient audit, numerical path integration, slope finite differences, directional HVP checks, acquisition audit, metrics and export are retained.
- Set OLMO_SOURCE to the original olmo_association_v1. PREVIOUS can point to paired_models_share.zip, multimodel_share.zip, the corresponding extracted directory, or frozen_predictors.json.
- Transformers 4.46.3 does not support Qwen3. The notebook offers an explicit Setup Qwen environment button: it creates a NEW venv using the current environment's CUDA Torch, installs Transformers 4.57.6 there, and does not upgrade the running experiment environment. It does not change the notebook kernel. Launch uses that environment's Python directly.
- Qwen sessions have an 11.5-hour budget with resumable saved work. Stop and wait for alive=False before using the GPU elsewhere.
- Export via the notebook writes qwen3_models_share.zip. Model/optimizer checkpoints remain local and are excluded from the share.

## OPEN-1B: important prerequisite on GH200

The ORIGINAL native training computation requires Gensyn's RepOps operators and replay dependencies. The published volunteer runbook documents Linux x86-64 and macOS ARM64 wheels; your GH200 host is likely Linux aarch64. This package cannot manufacture a compatible proprietary/native binary. Use a compatible official build in an isolated environment, or run the native stage on a supported x86-64 NVIDIA host. The platform itself is not hard-blocked if working operators are available.

Set OPEN_PYTHON to the Python executable in that native environment. Check OPEN readiness before downloading the checkpoint. If unavailable, status is dependency_blocked; Qwen can still run independently. There is deliberately NO substitution of fresh optimizer state or HF inference gradients for original pretraining history.

Official setup/runbook: https://github.com/gensyn-ai/open-transformers/blob/d7b7b674cd851f7a16815229f060658180cb772b/scripts/audit_volunteer/RUNBOOK.md

Use the official matching audit kit and dependencies; do not install them into the Qwen environment. This repository's pretrain dependencies include an older tokenizers pin, another reason to keep environments separate. Compatible native kernels may still fail replay verification; a library import alone is not proof of compatibility.

## What the OPEN pilot actually does

1. Loads ONE official checkpoint at pretraining step 50,300 (about 235B tokens), chosen before looking at its effects. Downloads only its files, approximately 19.3 GB, not the trajectory. Object generations and published checksums are recorded/verified. It fetches the needed native training data shards through the original replay harness.
2. Replays ONE actual next pretraining update by default; settings allow at most three. A single update processes the original large global batch (~4.7M tokens at this checkpoint), so one step can take substantial time on one GPU. No 10-hour completion guarantee is made.
3. Loads the original weights, optimizer moments, per-parameter clocks, data stream, schedule and clipping state through the native loader. Checks optimizer coverage, nonzero historical moments, finite states and positive clocks. No order-based transplant of optimizer tensors into a different model.
4. Observes native AdamW history/current numerator channels with the same updated denominator, and separately records weight decay, optimizer-formula rounding discrepancy, and post-optimizer scale refresh/clamps. The checkpoint refreshes LSQ scales every step: this is not mislabeled as floating-point roundoff. Skipped updates are explicitly labeled; their hypothetical h/c are not attributed to the actual displacement.
5. Runs small diagnostic CE probes at alpha=0,0.5,1 along the realized parameter displacement. Checks the native operator autograd directional derivative against central finite differences at multiple widths. This is a pilot derivative gate; quantized/straight-through derivatives need not equal the derivative of the realized forward. It does NOT assume a smooth Hessian for the quantized computation or certify a path integral from three samples.
6. Restores observer-modified state and requires the final canonical state hash to match the published target. If it does not, results remain unverified. Instrumentation uses four exact source insertion points in a pinned replay file and rejects unexpected layouts. Observer code runs after global gradient accumulation and clipping; it does not replace the real optimizer.
7. Saves one analysis capture per replayed step (before weights, realized delta, h, c) for separate audits. This can be ~26 GB per step; reserve at least 140 GB free for the default pilot, plus potentially additional fetched data shards. The observer also keeps several full-size CPU tensor copies; plan for roughly 128 GB or more host RAM. The official low-memory replay claim does not apply to this additional instrumentation. No ongoing full gradient history is saved.
8. AFTER native replay verification, run Proxy audits using QWEN_PYTHON. This loads the same checkpoint tensors and native displacement into the pinned HF architecture with an exact name/shape mapping. It compares (a) the quantized HF inference forward/autograd and (b) an FP32 unquantized alternative. All learned scale parameters are retained in the state, even though the FP32 alternative does not use them. The pinned inference source expects a newer Transformers mask-helper API; this probe adapter uses explicit uncached/unpadded causal and 32-token-block-aligned window masks matching its documented rule, and math SDPA for second derivatives. No cached-generation comparison is implied. This changes the computation and is clearly labeled as a surrogate. HVP curvature is attempted only for the FP32 alternative after its first-derivative gate passes.

The HF quantized port reproduces LSQ-style linear quantization but not the full native attention kernels or original training backward. It is never called the native quantized training model. Native evaluation CE is also kept separate from the original training CE+z-loss used to produce the update.

HF source pin: Gensyn/open-1b-base @ c76f21681ae429e3bc2d3978cd07796ccf62229a
Training/replay source pin: gensyn-ai/open-transformers @ d7b7b674cd851f7a16815229f060658180cb772b
Model card: https://huggingface.co/Gensyn/open-1b-base

## What this pilot can and cannot establish

The pilot tests feasibility of faithful historical-state loading, real-history channel accounting, and derivative validity. Its default short prose probe is a numerical smoke probe, NOT an established retention benchmark, and is not certified absent from pretraining. It cannot by itself prove forgetting of previously learned knowledge. For a subsequent substantive study, supply and pre-register meaningful fixed retention probes (`probe_token_ids`) with provenance, disjoint from the continuation batches, and enough evaluation coverage. The scope here is deliberately the requested short pilot before scheduling that study.

FD settings are recorded: widths [0.02,0.01,0.005,0.0025] in fractions of the actual update, atol=0.002 and rtol=0.1. Native/FP derivative gates require agreement with autograd at two adjacent widths and agreement between those finite differences. They are screening tolerances, not a global differentiability certificate. The FP curvature checks are local and conditional; no full Hessian spectrum or PH is claimed.

Native autograd can fail on checkpointed/FSDP/custom operators. Such exceptions are recorded as failed/unsupported derivative audits; the canonical replay may still verify. Completion and numerical validity are separate. No automatic full OPEN study is launched.

## Controls and restart behavior

- Refresh reads saved status. A live process does not imply recent progress; long replay/setup operations may be between pulses.
- Qwen stop resumes completed acquisition/trajectory/path work.
- OPEN stop checks during downloads, microbatches and diagnostic work. It may wait for a native operation to finish. An INTERRUPTED native interval restarts from the original checkpoint; the pilot does not fabricate a partial canonical checkpoint. Completed verified native replay is not rerun on Launch.
- Proxy audits resume completed per-step JSON files.
- Scientific settings changes require a new OPEN output directory. Never delete the original checkpoint or old research outputs to restart.
- GPU guards cover the directories listed in OTHER_GPU_RUNS and the two new outputs. Add any other active studies; this is not a machine-wide scheduler.
- Native export includes reports/provenance/diagnostics/code, not checkpoint/data assets or large capture tensors. Keep local captures for reproducibility.

See VALIDATION.txt for what was actually tested locally and what remains untested.
