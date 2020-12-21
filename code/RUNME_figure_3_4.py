# -*- coding: utf-8 -*-
"""
This script reproduces all MR images in Figure 3 and 4 of: L. Bungert and M. J. Ehrhardt, 
"Robust Image Reconstruction with Misaligned Structural Information," 
in IEEE Access, doi: 10.1109/ACCESS.2020.3043638.

The script mri_multistep_method.py requires a version of MATLAB which is compatible 
with your python version.
"""
exec(open("mri_affreg_multigrid.py").read())
exec(open("mri_multistep_method.py").read())

