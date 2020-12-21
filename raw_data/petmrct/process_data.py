import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat as sp_loadmat

input_folder = '.'
output_folder = '../../data'

input_fnames = ['data1', 'data2', 'data3']
input_format = 'png'

data_name = 'pet'
sinfo_name = 'mr'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

for input_fname in input_fnames:
    data = plt.imread('{}/{}_{}.{}'.format(
            input_folder, input_fname, data_name, input_format))
    sinfo = plt.imread('{}/{}_{}.{}'.format(
            input_folder, input_fname, sinfo_name, input_format))
    
   
    data = data[7:127,10:130,]
    data = rgb2gray(data)
    sinfo = sinfo[7:127,10:130,]
    sinfo = rgb2gray(sinfo)
    
    data /= data.max()
    sinfo /= sinfo.max()
    
    output_fname = input_fname
    
    prefix = '{}/{}'.format(output_folder, output_fname)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    np.save(prefix, (data, sinfo))
    plt.imsave(prefix + '_sinfo.png', sinfo, cmap='gray')
    plt.imsave(prefix + '_data.png', data,   cmap='inferno')
