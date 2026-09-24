# Complete OLMo structure study

1. Extract all files into a new folder on your GPU machine.
2. Open Run_structure_study.ipynb using your existing OLMo Python environment.
3. Choose Run All. Rerun the Refresh cell for status, numbers, and plots.

No files from an older code package need to be copied. The original model checkpoints remain external and are discovered at /home/ubuntu/1/runs/olmo_association_v1 or nearby. If needed, set SOURCE in the first cell to the original campaign folder. This package does not contain model checkpoints.

The default action resumes. For a fresh run change ACTION to "restart" and run Start. For stop use "stop". Outputs are preserved, and the actual output folder is remembered in .olmo_structure_session.json beside the notebook. Use one notebook session per folder. A running older configuration must be stopped/restarted before changed settings take effect.

Included: the full structure_study.py implementation, full forgetting_mechanisms.py experiment core, notebook_runner.py entry points, notebook, requirements, methods, and validation notes. The existing GPU environment must provide the listed dependencies; this notebook does not install packages or modify your environment automatically.

The default evaluates seed 1. Extend seeds in OPTIONS for other available seeds. METHODS.md documents optional real pretraining checkpoints and all geometry/topology measures. A random model and a final pretrained model do not reveal the actual pretraining path. Structural associations do not establish causes of forgetting.
