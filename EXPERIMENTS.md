# Experiment catalogue

Suite folders preserve independent imports and environments. Do not combine their Python modules in one directory. Read each local README before execution.

| Directory under `suites/` | Purpose | Notebook(s) |
|---|---|---|
| forgetting_mechanism_suite | Exploratory forgetting mechanisms | `Run_forgetting_mechanisms.ipynb` |
| olmo_structure_study | Initial weight and activation structure | `Run_structure_study.ipynb` |
| olmo_structure_complete | Restartable structure study | `Run_structure_study.ipynb` |
| olmo_structure_ph_rmt_ntk | Persistent homology, RMT and task-coordinate NTK | `Run_structure_study.ipynb` |
| olmo_task_geometry | Native Adam task-aware update geometry | `Run_structure_study.ipynb` |
| olmo_update_path | Within-update paths | `Run_structure_study.ipynb` |
| olmo_overnight | Initial overnight path and branch study | `Run_structure_study.ipynb` |
| olmo_mechanism_bridge | Channel factorials and activation interventions | `Run_mechanism_bridge.ipynb` |
| olmo_bridge_speedup | Optional bridge storage/performance patch | `Apply_speedup.ipynb` |
| olmo_natural_geometry_overnight | Natural geometry trajectories | `Run_Natural_Geometry.ipynb` |
| olmo_precise_forgetting_overnight | Matched prompt/entity derivatives | `Run_Precise_Forgetting.ipynb` |
| olmo_path_refinement_overnight | Adaptive natural-path refinement | `Run_Path_Refinement.ipynb` |
| olmo_four_vector_study | Original OLMo four-vector trajectories | `run_four_vector_study.ipynb` |
| multimodel_forgetting_study | Pythia and original multimodel registry replication | `run_paper_suite.ipynb` |
| forgetting_validation | Retention and functional breadth | `Run_retention_analysis.ipynb` |
| frame_history_study | OLMo frame-by-history factorial | `Run_frame_history.ipynb` |
| paired_base_instruct_study | Qwen2.5 and SmolLM2 base/instruct registry comparison | `Run_paired_models.ipynb` |
| qwen3_open1b_study | Original Qwen3 registry; optional OPEN-1B pilot code (not evidence of completed pilot) | `Run_Qwen3_and_OPEN.ipynb` |
| overnight_replication_study | Fresh-seed text and intervention replication | `Run_Overnight_Replication.ipynb` |
| optimizer_generality_study | Timing, optimizer calibration, answer-token controls | `Run_Optimizer_Generality.ipynb` |
| history_geometry_followup | Matched-start history branches and persistence controls | `Run_History_Geometry.ipynb` |
| natural_forgetting_components_fixed | Natural confusion/leakage channels and finite-path Hessians; output-head fix | `Run_Natural_Components.ipynb` |
| history_dynamics_replay | Seven-trajectory history-age and cross-time analysis | `Run_History_Dynamics.ipynb` |
| qwen3_history_replication | Ongoing Qwen3 text/history/cohort/finite-path replication | `Run_Qwen3_History.ipynb` |
| reviewer_experiment_suite | Prediction policies, history/LR controls, frame replication and SST-2 to AG News | `Run_Reviewer_Experiments.ipynb` |
