# Robust Image Reconstruction with Misaligned Structural Information
This Python code allows to reproduce the results of <b>Robust Image Reconstruction with Misaligned Structural Information</b> [1].

[1] L. Bungert and M. J. Ehrhardt (2020). Robust Image Reconstruction with Misaligned Structural Information. IEEE Access. https://doi.org/10.1109/ACCESS.2020.3043638.

The aim of [1] is to reconstruct an image from an indirect measurement whilst registering it with a structural side information from a different modality.

## Prerequistes
Our code requires the Operator Discretization Library (ODL) https://odlgroup.github.io/odl/index.html.
Furthermore, the scripts [mri_multi_step_method.py](code/mri_multi_step_method.py),
[ct_multi_step_method.py](code/ct_multi_step_method.py),
and [ct_rotations_multi_step_method.py](code/ct_rotations_multi_step_method.py),
which implement a three-step reconstruction and registration method call MATLAB 
which requires version compatibility (https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-support.pdf)
and an installation of the MATLAB engine (https://de.mathworks.com/help/matlab/matlab-engine-for-python.html).
Since the three-step method only serves as comparison for our proposed method,
these scripts are not essential.


## Getting started
There are a number of scripts which reproduce the results as presented in the paper. 
First you have to unpack the raw data by running process_data.py in every subfolder of [raw_data](raw_data).

To produce all images involved in the figures of [1], simply run the respective scripts
in [code](code), for instance, [RUNME_figure_3_4.py](code/RUNME_figure_3_4.py).

This calls the following scripts which can be called on their own and
can be adapted to individual purposes:

* [mri_affreg_multigrid.py](code/mri_affreg_multigrid.py)
* [mri_multi_step_method.py](code/mri_multi_step_method.py)
* [ct_affreg_multigrid.py](code/ct_affreg_multigrid.py)
* [ct_multi_step_method.py](code/ct_multi_step_method.py)
* [ct_rotations_affreg_multigrid.py](code/ct_rotations_affreg_multigrid.py)
* [ct_rotations_multi_step_method.py](code/ct_rotations_multi_step_method.py)
* [hs_affreg_multigrid.py](code/hs_affreg_multigrid.py)

## References
[1] L. Bungert and M. J. Ehrhardt (2020). Robust Image Reconstruction with Misaligned Structural Information. IEEE Access. https://doi.org/10.1109/ACCESS.2020.3043638.

