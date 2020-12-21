# -*- coding: utf-8 -*-
"""
This script reproduces all CT images in Figures 5 and 6 of: 
L. Bungert and M. J. Ehrhardt, "Robust Image Reconstruction with Misaligned 
Structural Information," 
in IEEE Access, doi: 10.1109/ACCESS.2020.3043638.

Note that our results were computed using the Astra Toolbox using the backend 
'astra_cpu'. You can change the backend of the Radon transform to 'astra_cuda' 
or 'skimage' in the respective scripts. However, parameters might have to be 
adapted. 

The script ct_multistep_method.py requires a version of MATLAB which is 
compatible with your python version.
"""
exec(open("ct_affreg_multigrid.py").read())
exec(open("ct_multistep_method.py").read())

