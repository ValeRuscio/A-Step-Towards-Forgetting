# Paper-to-code map

Use section topics rather than table numbers, which change during manuscript editing. The current manuscript's central text replay is the seven-trajectory Pythia/Smol snapshot. New Qwen text results must be labeled separately until their analyses complete.

| Claim / experiment | Distributed suite under `suites/` | Recorded provenance under `recorded_runs/` |
|---|---|---|
| Natural confusion/leakage, history/current projection, age bins, temporal vector and norm/cosine decompositions | `natural_forgetting_components_fixed` generates source; `history_dynamics_replay` performs replay | `history_dynamics_share` contains replay code/settings and source environment/settings; full original natural source database must be supplied separately |
| New Qwen3 text replication: seeds 61–63, shared/disjoint, age norms/cosines, fixed-gradient cohorts, deferred finite paths | `qwen3_history_replication` | No completed exported run archive available in this release; distributed source only |
| Finite-update gradients, Hessian channel coupling, residual accounting, uniform vs enriched path selection | `natural_forgetting_components_fixed` | No complete exported natural-components run code snapshot available here; preserve final remote export |
| Original OLMo full four-vector trajectories and frozen prediction | `olmo_four_vector_study`, `multimodel_forgetting_study` | `four_vector_share`, `multimodel_share` |
| Refined OLMo diagnostic paths | `olmo_precise_forgetting_overnight`, `olmo_path_refinement_overnight` | Dependent source inputs described in suite READMEs; no separate recorded source snapshot from their result-only archives |
| Eight-checkpoint registry comparison and selected paths | `multimodel_forgetting_study`, `paired_base_instruct_study`, `qwen3_open1b_study` | `multimodel_share`, `paired_models_share`, `qwen3_models_share` |
| OLMo held-out frame × channel factorial / rotations | `frame_history_study` | `frame_history_v1_share_latest` |
| Fresh-seed frame replication, policies, LR/history controls and text extension | `reviewer_experiment_suite`, `overnight_replication_study` | `reviewer_results_share`, `overnight_replication_share` |
| Timing/persistence, SGD/momentum/Adam beta1=0/Adam, calibration and answer-token controls | `optimizer_generality_study` | `optimizer_generality_share` |
| Seeds 41–43 matched-start norm-controlled branches, frozen predictors and paired paths | `history_geometry_followup` | Only an unverified-association loose settings file is available here; no executed-code snapshot was supplied with the loose summaries |
| Retention/recovery and functional breadth | `forgetting_validation` | Requires original saved trajectory inputs; see local README |
| Exploratory weight/activation spectra, PH/RMT/NTK, bridge interventions | Structure, task-geometry, overnight and mechanism-bridge suites in catalogue | Exploratory packages; distinguish from the paper's primary natural-history evidence |

## Suggested reproduction order

For the main text mechanism: generate natural source trajectories with the fixed natural-components suite; retain its settings, environment, data directory and SQLite records; then run history replay on an idle source. Replay freezes the completed cases at snapshot time. A seven-case snapshot does not automatically grow into a twelve-case analysis.

For Qwen text replication: use its standalone notebook; the pipeline sequences natural training, exact history replay, then selected paths. It intentionally differs from the full Pythia/Smol path budget. Do not pool its sampled-path counts without recording those selection differences.

For intervention/forecast evidence: follow the named suite, keeping its own frozen predictors, development records and fresh-seed settings. Default settings and the actual settings exported by a completed run may differ; the latter define the reported run.
