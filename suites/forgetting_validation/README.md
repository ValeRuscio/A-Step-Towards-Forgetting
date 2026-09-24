# Consequential forgetting follow-up

This package contains a completed read-only analysis of existing four-vector results and a proposed GPU follow-up protocol. It does NOT launch new GPU experiments or change the currently running multimodel suite.

Open Run_retention_analysis.ipynb, set SOURCE to the original four-vector result directory or share ZIP, and run the cell. Only the Python standard library is required. Or run:

python analyze_retention.py /path/to/four_vector_share.zip --output retention_analysis

Included results/ were generated from the provided four_vector_share.zip. REPORT.txt summarizes four previously selected events. all_updates.csv covers all 384 updates; entity_changes.csv retains entity effects. The analysis defines positive change using 1e-4 nats and recovery as eight consecutive endpoints at or below the pre-event level plus that tolerance. These are descriptive retrospective settings, not preregistered thresholds.

The restricted accuracy compares the two answer logits, with half-credit ties. It is NOT full-vocabulary top-1 accuracy. Majority-answer probability is reconstructed from their logit margin and negative log total answer mass. Target probabilities are soft (.9/.1), so majority-answer accuracy is complementary to KL, not the objective itself.

EXPERIMENT_PROTOCOL.md separates natural geometry/topology measurement from causal branch and activation tests, identifies controls and future-data selection rules, and records limits of the current evidence. New GPU code for those interventions is not included in this package; existing measurements are reusable now.

Validation: standard-library tests pass for probability reconstruction, tie handling, recovery and censored follow-up. The analyzer was run successfully on all 384 saved updates. Original archives and experiment code were not modified.
