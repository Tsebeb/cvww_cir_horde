# Horde Strategy for Class Incremental Learning with Repetition 

Our paper "Incremental Learning with Repetition via Pseudo-Feature Projection" has been accepted at the 
28th Computer Vision Winter Workshop (CVWW) at Graz. 

This repository contains the official reference implementation.
The code structure is based on [FACIL](https://github.com/mmasana/FACIL).

### Experiments
To run the experiments for our paper please run the corresponding scenario python files under "run_scripts".
For example to reproduce the results for the base EFCIR scenario b, run:

    python scenario_b_efcir.py

The CIFAR dataset for the experiment is automatically downloaded from torchvision.

### Citation
If you find this works interesting, please cite us with the following

    @inproceedings{tscheschner2025efcir,
                   title={Incremental Learning with Repetition via Pseudo-Feature Projection},
                   author={Tscheschner, Benedikt and Veas, Eduardo and Masana, Marc},
                   journal={Computer Vision Winter Workshop},
                   doi={10.3217/978-3-99161-022-9-004},
                   year={2025}}
