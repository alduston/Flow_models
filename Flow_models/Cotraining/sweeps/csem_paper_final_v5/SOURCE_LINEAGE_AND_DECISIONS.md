# V4 source decisions

V3 selected a canonical oracle-initialized FMNIST experiment to reproduce a Gaussian-start headline. V4 replaces FMNIST main with the closest recovered deployable terminal-KL experiment, retaining its archived joint trainer instead of transplanting hyperparameters into a changed two-horizon trainer. All source configuration values were compared against an intercepted execution of the recovered CLI.

This does not establish an exact 5.04 reproduction: the recovered Gaussian log reaches Heun5.13/RK4 final5.24, whereas the separate oracle log reaches4.97. The older paper5.03 is a third experiment.

The modern CIFAR co-training route is retained. Modern independent-pair tracking input, tracking LR, evaluation, and compiler selection bugs are repaired. Detailed evidence and rerun instructions are in `FMNIST_CONFIGURATION_AUDIT.md`. `PATCH_FROM_V3.diff` records the source changes.
