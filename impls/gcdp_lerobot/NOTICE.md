# Goal-conditioned Diffusion Policy compatibility code

`configuration_diffusion.py` and `modeling_diffusion.py` are derived from
Hugging Face LeRobot commit `e86f5af5` plus the goal-conditioning changes used
to train the GC-DP checkpoint. They are included only to load that checkpoint
without replacing the repository's pinned `lerobot==0.4.3` installation.

The original LeRobot source is licensed under the Apache License 2.0. Copyright
notices and license headers are preserved in both derived files.
