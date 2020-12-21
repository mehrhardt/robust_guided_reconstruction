import numpy as np
import odl
import functionals as fctls
import algorithms as algs
import operators as ops
import deform as defm
import matplotlib.pyplot as plt
import skimage.transform as sktransform
from skimage.metrics import structural_similarity as ssim
import os
import matlab.engine


#%%
# Model parameters
stddev      = 0.05              # standard deviation for gaussian noise
num_angles  = 200               # number of Radon projection angles
phis         = [0.2, 0.4, 0.6]  # rotation angles between side info and data
shift       = [0.02, 0.08]      # shift between side info and data
noiseModel  = 'poisson'         #'gaussian', 's-p', 'poisson'
noise       = True

# Algorithmic parameters
maxIt       = 20000         # maximal iterations for black-box registration
save2Disk   = True
niter       = [10, 20, 50, 100, 200, 500]   # number of PALM iterations
data_aligned = False
use_sinfo   = True
NonNeg      = True
impl        = 'astra_cpu'  


# Regularization and vectorfield parameters
alphas =  [4e-2, 4e-1]
gamma = 0.9995
eta = 1e-2

data_folder = '../data'
data_fnames = ['data3']


output_folder = '../results'

#  start matlab engine
eng = matlab.engine.start_matlab()

for data_fname in data_fnames:
    for phi in phis:
        folder_out = '{}/{}'.format(output_folder, data_fname) 
        folder_out += '_angles_{}'.format(num_angles)
        # folder_out += '_shift_{}-{}_rot_{}'.format(shift[0],shift[1],phi)
        folder_out += '_rot_{}'.format(phi)
        folder_out += '_noiseModel_{}'.format(noiseModel) 
        folder_out += '_MS_masked'
        
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)
            
        # Target space
        gt, sinfo = np.load('{}/{}.npy'.format(data_folder, data_fname))
        gt = np.pad(gt, ((12, 12), (12, 12)), 'edge')
        sinfo = np.pad(sinfo, ((12, 12), (12, 12)), 'edge')
        simage = sinfo.shape
        X = odl.uniform_discr([-1, -1], [1, 1], simage, interp='linear')
        V = X.tangent_bundle
        Yaff = odl.tensor_space(6)
        Z = odl.ProductSpace(X, Yaff) 
        gt = X.element(gt)
        sinfo = X.element(sinfo)
        
        # Generate affine map and data image
        cosp = np.cos(phi)
        sinp = np.sin(phi)
        disp_func = [
            lambda x: (cosp-1)*x[0] - sinp*x[1] + shift[0],
            lambda x: sinp*x[0] + (cosp-1)*x[1] + shift[1]]

        vf_gt = Yaff.element([shift[0], shift[1], cosp-1, -sinp, sinp, cosp-1])
        
        vf = V.element(disp_func)
        
        X_plot = odl.uniform_discr([-1, -1], [1, 1], (15, 15), interp='linear')
        V_plot = X_plot.tangent_bundle
        
        x,y = X_plot.meshgrid
        u = disp_func[0]([x,y])         
        v = disp_func[1]([x,y]) 
        
        plt.quiver(x,y,u,v,color='yellow')
        plt.axis('equal'),
        plt.axis([-1.3,1.3,-1.3,1.3]),
        plt.gca().set_aspect('equal', adjustable='box'),
        plt.savefig(folder_out + '/deformation_field.png',
                    bbox_inches='tight', 
                    pad_inches=0, 
                            transparent=True, dpi=600,format='png')
                
        # Create data
        angle_partition = odl.uniform_partition(0, np.pi, num_angles)
        detector_partition = odl.uniform_partition(-1.5, 1.5, int(1.6 * sinfo.shape[0]))
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
        A = odl.tomo.RayTransform(X, geometry)#, impl='astra_cpu')
        FBP = odl.tomo.fbp_op(A, filter_type='Hann')
        
    
        deform_op = defm.LinDeformFixedDisp(vf)
        deformed_gt = deform_op(gt)
        
        
        deformed_sino = A(deformed_gt)
        aligned_sino = A(gt) 
        poisson_factor = 100/np.max(aligned_sino)
        if noiseModel == 'poisson':
            A *= poisson_factor
        
        
        aligned_data = aligned_sino
        deformed_data = deformed_sino
        
        if noise:
            if noiseModel == 'gaussian':
                noise = odl.phantom.white_noise(A.range, mean=0, stddev=stddev, seed=1)
                aligned_data += noise
                deformed_data += noise
            elif noiseModel == 's-p':
                aligned_data = odl.phantom.salt_pepper_noise(aligned_data, seed=1)
                deformed_data = odl.phantom.salt_pepper_noise(deformed_data, seed=1)
            elif noiseModel == 'poisson':
                background =  7 * aligned_sino.space.one()
                aligned_data = odl.phantom.poisson_noise(poisson_factor*aligned_sino + background, seed=1807)
                deformed_data = odl.phantom.poisson_noise(poisson_factor*deformed_sino + background, seed=1807)
            else:
                raise NotImplementedError('Unknown or missing noise type')
        else:
            if noiseModel == 'poisson':
                background =  7 * aligned_sino.space.one()
                aligned_data = poisson_factor * aligned_sino  + background
                deformed_data = poisson_factor * deformed_sino + background       
        
       
        # Save data images
        plt.imsave(folder_out + '/' + data_fname + '_data.png', aligned_data, cmap='gray')
        plt.imsave(folder_out + '/' + data_fname + '_data_deformed.png', deformed_data, cmap='gray')
        plt.imsave(folder_out + '/' + data_fname + '_gt_deformed.png', deformed_gt, cmap='inferno')
        plt.imsave(folder_out + '/' + data_fname + '_gt.png', gt, cmap='inferno')
        plt.imsave(folder_out + '/' + data_fname + '_fbp_deformed.png', FBP(deformed_data), cmap='inferno')
        plt.imsave(folder_out + '/' + data_fname + '_fbp.png', FBP(aligned_data), cmap='inferno')
        plt.imsave(folder_out + '/' + data_fname + '_sinfo.png', sinfo, cmap='gray')
        
        #%% reconstruction
        im_init = X.zero()
        vf_init = Yaff.zero()
      
               
        # Set some parameters and the general TV prox options
        prox_options = {}
        prox_options['name'] = 'FGP'
        prox_options['warmstart'] = True
        prox_options['p'] = None
        prox_options['tol'] = None
        prox_options['niter'] = 10  #100
        strong_convexity = 0
        
        # Define non-smooth part
        reg_affine = odl.solvers.ZeroFunctional(Yaff)   
           
        step = 10
        if data_aligned is True:
            data = aligned_data
        else:
            data = deformed_data
        
               
        if noiseModel == 'gaussian':
            datafit = 0.5 * odl.solvers.functional.L2NormSquared(data.space).translated(data)
        elif noiseModel == 's-p':
            datafit = odl.solvers.functional.LpNorm(data.space, 1).translated(data)   
        elif noiseModel == 'poisson':
            datafit = odl.solvers.functional.KullbackLeibler(data.space, prior=data).translated(-background)
        else:
            raise NotImplementedError('Unkown noise type')
            
        f = fctls.DataFitDisp(Z, datafit, forward=A)
                  
    
        for it in range(2):
            
            # reconstruct only image
            RegParam = alphas[it]
            if it == 0:
                reg_im = fctls.TV(X, alpha=RegParam, sinfo=None, 
                                  NonNeg=NonNeg, 
                                  prox_options=prox_options.copy(), 
                                  gamma=gamma, eta=eta)
            else:
                reg_im = fctls.TV(X, alpha=RegParam, sinfo=sinfo, 
                                  NonNeg=NonNeg, 
                                  prox_options=prox_options.copy(), 
                                  gamma=gamma, eta=eta)
            
            ud_vars = [0]
            file_out = 'recon_{}_da{}_si{}_vars{}_a{:4.2e}_g{:6.4f}_e{:6.4f}'.format(it, 
                     data_aligned, use_sinfo, ud_vars, RegParam, gamma, eta)
                            
            g = odl.solvers.SeparableSum(reg_im, reg_affine) 
        
            # Define objective functional
            obj = f + g
            
            x_init = Z.element([im_init,vf_init])
         
            cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                  odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                  odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True) &
                  odl.solvers.CallbackPrint(step=step, func=obj, fmt='obj={!r}') & 
                  odl.solvers.CallbackShow(step=step))
        
            recon = algs.PALM(f, g, ud_vars=ud_vars.copy(), x=x_init, niter=None, 
                                 callback=cb, L=None, tol=1e-9)    
        
        
            niter_diff = [niter[0]] + list(np.diff(niter)) 
          
            clim = [0, 1] 
            proj = odl.solvers.IndicatorBox(X, lower=clim[0], upper=clim[1]).proximal(1)
            
            for ni, nid in zip(niter, niter_diff):
                recon.run(nid)   
                if save2Disk is True:             
                    file_out_ni = '{}_{:04d}'.format(file_out, ni)            
                    plt.imsave(folder_out + '/' + file_out_ni + '.png', 
                               proj(recon.x[0]), cmap='inferno')
                    np.save('{}.npy'.format(folder_out + '/' + file_out_ni), list(recon.x))
            
            if it == 0:
                #register side information
                moving = proj(recon.x[0])         
                fixed = sinfo                            
                
                moving_matlab = np.array(moving).tolist()
                moving_matlab = matlab.double(moving_matlab)
                
                fixed_matlab = np.array(fixed).tolist()
                fixed_matlab = matlab.double(fixed_matlab)
                
               
                affmap = eng.compute_affine_map(moving_matlab, fixed_matlab, maxIt)    
                affmap = np.array(affmap)
                vf_registered = Yaff.element([affmap[2,1], affmap[2,0],
                                              affmap[0,0]-1, affmap[0,1],
                                              affmap[1,0], affmap[1,1]-1,
                                              ])
                
                rel_diff_vf = (vf_registered-vf_gt).norm()/vf_gt.norm()
                
                vf_init = vf_registered
                        
                
        ssim_val = ssim(recon.x[0], gt)        
    
        if save2Disk is True:
            file1 = open("{}.txt".format(folder_out + '/' + file_out_ni),"w")   
            file1.write("SSIM: {} \n".format(ssim_val)) 
            file1.write("RD Vf: {} \n".format(rel_diff_vf)) 
            file1.close()
         