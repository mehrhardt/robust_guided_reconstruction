import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat as sp_loadmat

input_folder = '.'
output_folder = '../../data'

input_format = 'mat'
input_fname = 'urban'

#input_format = 'bsq'
#input_filename = 'hyp_sample03'
#sinfo_filename = 'rgb_sample03.tif'

data_obj = sp_loadmat('{}/{}.{}'.format(input_folder, input_fname, input_format))

data = data_obj['MS']
sinfo = data_obj['PAN']

ichannel = 3
fov = [0, 100, 100, 200]
output_fname = '{}_ch{}_{}-{}x{}-{}'.format(input_fname, ichannel,
                                               fov[0], fov[1], fov[2], fov[3])

sdata = data.shape[:2]
s = 4
sinfo = sinfo[fov[0] * s:fov[1] * s, fov[2] * s:fov[3] * s]

data = data[fov[0]:fov[1], fov[2]:fov[3], ichannel]

data /= data.max()
sinfo /= sinfo.max()

prefix = '{}/{}'.format(output_folder, output_fname)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

np.save(prefix, (data, sinfo))
plt.imsave(prefix + '_sinfo.png', sinfo, cmap='gray')
plt.imsave(prefix + '_data.png', data, cmap='nipy_spectral')
plt.imsave(prefix + '_data.png', data, cmap='gist_ncar')
#plt.imsave(prefix + '_data.png', data, cmap='terrain')
#plt.imsave(prefix + '_data.png', data, cmap='YlGnBu')




