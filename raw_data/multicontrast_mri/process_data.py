import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat as sp_loadmat

input_folder = '.'
output_folder = '../../data'

input_fnames = ['BrainWebA', 'BrainWebB', 'BrainWebC', 'phantom', 'patientA', 'patientB']
input_format = 'mat'

for input_fname in input_fnames:
    data_obj = sp_loadmat('{}/data_{}.{}'.format(
            input_folder, input_fname, input_format))['groundtruth'][0][0]
    
    for ichannel in range(2):
        data = data_obj[ichannel]
        sinfo = data_obj[np.mod(ichannel+1, 2)]
        
        output_fname = 'mri_{}_ch{}'.format(input_fname.lower(), ichannel)
        
        data /= data.max()
        sinfo /= sinfo.max()
        
        prefix = '{}/{}'.format(output_folder, output_fname)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        np.save(prefix, (data, sinfo))
        plt.imsave(prefix + '_sinfo.png', sinfo, cmap='gray')
        plt.imsave(prefix + '_data.png', data, cmap='gray')
