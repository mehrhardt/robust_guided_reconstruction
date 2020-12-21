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

#%%
# Model parameters
stddev      = 0.0015                # standard deviation for gaussian noise
num_angles  = 200                   # number of Radon projection angles
phis        = [0.2, 0.4, 0.6, 0.8, 1, 1.2]  # rotation angles between side info and data
shift       = [0.02, 0.08]          # shift between side info and data
noiseModel  = 'poisson'             # 'gaussian', 's-p', 'poisson'
noise       = True

# Algorithmic parameters
save2Disk   = True
niter       = [10, 20, 50, 100, 200, 500]   # number of PALM iterations
resolutions = [9, 18, 36, 72, 144] 
factor      = 2
data_aligned = False
use_sinfo   = True
ud_vars     = [0, 1]
tol         = 1e-9
impl        = 'astra_cpu'  

# Regularization and vectorfield parameters
alphas = [4e-1] 
gamma = 0.9995
eta = 1e-2

data_folder = '../data'
data_fnames = ['data3']


output_folder = '../results'

for data_fname in data_fnames:
    for alpha in alphas:
        for phi in phis:
            folder_out = '{}/{}'.format(output_folder, data_fname) 
            folder_out += '_angles_{}'.format(num_angles)
            folder_out += '_rot_{}'.format(phi)
            folder_out += '_noiseModel_{}'.format(noiseModel) 
            # folder_out += '_noise_{}'.format(noise) 
            folder_out += '_scale_space_{}'.format(len(resolutions)) 
            folder_out += '_multigrid'
            
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
            
            deform_op = defm.LinDeformFixedTempl(gt)
            
            # Create data
            angle_partition = odl.uniform_partition(0, np.pi, num_angles)
            detector_partition = odl.uniform_partition(-1.5, 1.5, int(1.6 * sinfo.shape[0]))
            geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
            A = odl.tomo.RayTransform(X, geometry, impl=impl)
           
            FBP = odl.tomo.fbp_op(A, filter_type='Hann')
    
            
            deformed_gt = deform_op(vf)
            deformed_sino = A(deformed_gt)
            aligned_sino = A(gt) 
            poisson_factor = 100/np.max(aligned_sino)
            
            
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
                
            
            file1 = open("{}.txt".format(folder_out + '/' + 'FBP'),"w")   
            if noiseModel == 'poisson':
                ssim_val_fbp = ssim((FBP(aligned_data)),gt)
            else:
                ssim_val_fbp = ssim(FBP(aligned_data),gt)
                
            file1.write("SSIM: {} \n".format(ssim_val_fbp))
            file1.close()
            
            
            
            # Save data images
            plt.imsave(folder_out + '/' + data_fname + '_data.png', aligned_data, cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_data_deformed.png', deformed_data, cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_gt_deformed.png', deformed_gt, cmap='inferno')
            plt.imsave(folder_out + '/' + data_fname + '_gt.png', gt, cmap='inferno')
            plt.imsave(folder_out + '/' + data_fname + '_fbp_deformed.png', FBP(deformed_data), cmap='inferno')
            plt.imsave(folder_out + '/' + data_fname + '_fbp.png', FBP(aligned_data), cmap='inferno')
            plt.imsave(folder_out + '/' + data_fname + '_sinfo.png', sinfo, cmap='gray')
            
            #%% reconstruction
            X_first = odl.uniform_discr([-1, -1], [1, 1], (resolutions[0],resolutions[0]), interp='linear')
            im_init = X_first.one()*1
            vf_init = Yaff.zero()
            
            
            ind = 0
            basis = 10
            alphas = [alpha*basis**i for i in range(len(resolutions))]
            alphas.reverse()
            
            for res in resolutions:
                RegParam = alphas[ind]
                
                simage_res = (res,res)
                if use_sinfo is True:
                    sinfo_res = sktransform.resize(sinfo, simage_res)
                else:
                    sinfo_res = None
                    
                X_res = odl.uniform_discr([-1, -1], [1, 1], simage_res, interp='linear')
                V_res = X_res.tangent_bundle
                Z_res = odl.ProductSpace(X_res, Yaff) 
                A_res = odl.tomo.RayTransform(X_res, geometry, impl=impl)
                if noiseModel == 'poisson':
                    A_res *= poisson_factor
                
            
            
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
                # reg_affine = odl.solvers.IndicatorBox(Yaff,-0.1,0.1)
                   
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
                    
                f = fctls.DataFitDisp(Z_res, datafit, forward=A_res)
        
                            
                file_out = '{}_da{}_si{}_vars{}_a{:4.2e}_g{:6.4f}_e{:6.4f}_res{}'.format(data_fname, 
                         data_aligned, use_sinfo, ud_vars, RegParam, gamma, eta, res)
                                
                reg_im = fctls.TV(X_res, alpha=RegParam, sinfo=sinfo_res, NonNeg=True, 
                          prox_options=prox_options.copy(), gamma=gamma, eta=eta)
                g = odl.solvers.SeparableSum(reg_im, reg_affine) 
            
                # Define objective functional
                obj = f + g
                
                x_init = Z_res.element([im_init,vf_init])
             
                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True) &
                      odl.solvers.CallbackPrint(step=step, func=obj, fmt='obj={!r}') & 
                      odl.solvers.CallbackShow(step=step))
            
                recon = algs.PALM(f, g, ud_vars=ud_vars.copy(), x=x_init.copy(), niter=None, 
                                     callback=cb, L=[1e2, 1e4], tol=tol)    
            
            
                niter_diff = [niter[0]] + list(np.diff(niter)) 
              
                clim = [0, 1] 
                proj = odl.solvers.IndicatorBox(X_res, lower=clim[0], upper=clim[1]).proximal(1)
                
                for ni, nid in zip(niter, niter_diff):
                    recon.run(nid)   
                    
                    vf =  ops.Embedding_Affine(Yaff, V_plot)(recon.x[1])
                    plt.quiver(x, y, vf[0], vf[1], color='red'),
                    plt.axis('equal'),
                    plt.axis([-1.3,1.3,-1.3,1.3]),
    #                        plt.axis('off'),
                    plt.gca().set_aspect('equal', adjustable='box'),
                    if save2Disk is True:             
                        file_out_ni = '{}_{:04d}'.format(file_out, ni)            
                        file_out_def_ni = '{}_def_{:04d}'.format(file_out, ni) 
                        plt.savefig(folder_out + '/' + file_out_def_ni + '.png', 
                                    bbox_inches='tight', 
                                    pad_inches=0, 
                                    transparent=True, 
                                    dpi=600, format='png')
                        plt.imsave(folder_out + '/' + file_out_ni + '.png', 
                                   proj(recon.x[0]), cmap='inferno')
                        np.save('{}.npy'.format(folder_out + '/' + file_out_ni), list(recon.x))
                
                
                if save2Disk is True:
                    rel_diff_vf = (recon.x[1]-vf_gt).norm()/vf_gt.norm()
                    file1 = open("{}.txt".format(folder_out + '/' + file_out_ni + '_def'),"w")   
                    file1.write("RelDiff VF: {} \n".format(rel_diff_vf)) 
                    file1.close()
                
                X_new = odl.uniform_discr([-1, -1], [1, 1], (2*res,2*res), interp='linear')
            
                S = ops.Subsampling(X_new, X_res, 0)
                im_init = S.adjoint(recon.x[0]).copy()
                vf_init = recon.x[1].copy()
                ind += 1
            
            if save2Disk is True:
                im_resc = recon.x[0]  
                
                ssim_val = ssim(im_resc,gt)
                rel_diff_vf = (recon.x[1]-vf_gt).norm()/vf_gt.norm()
                
                file1 = open("{}.txt".format(folder_out + '/' + file_out_ni),"w")   
                file1.write("SSIM: {} \n".format(ssim_val)) 
                if use_sinfo is False:
                    ssim_val_def = ssim( im_resc, deformed_gt)
                    file1.write("SSIM def: {} \n".format(ssim_val_def))
                file1.write("RelDiff VF: {} \n".format(rel_diff_vf)) 
                file1.close()
            
            
            
             