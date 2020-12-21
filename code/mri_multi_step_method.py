import numpy as np
import odl
import functionals as fctls
import algorithms as algs
import operators as ops
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import deform as defm
import os
import matlab.engine

#%%
# Model parameters
stddev      = 0.001          # standard deviation for gaussian noise
phi         = 0.1
shift       = [-0.02, -0.08]         
zoomfactor  = 0.85
deformation = 'zoom_rigid' #'zoom', 'rigid', 'mixed'
NonNeg      = False
unitary     = False

# Algorithmic parameters
maxIt       = 20000         # maximal iterations for black-box registration
save2Disk   = True
niter       = [10, 20, 50, 100, 200, 500]   # number of PALM iterations
data_aligned = False
use_sinfo   = True

# Regularization and vector field parameters
alphas = [1e-4, 5e-3]
gamma = 0.9995
eta = 1e-2

data_folder = '../data'
data_fnames = ['mri_patienta_ch0'] #['mri_brainweba_ch0', 'mri_phantom_ch0']
samplings =  ['radialbl-15'] #['cartesian-2', 'cartesian-4','random-0.5','radial-70','radialbl-30',]


output_folder = '../results'

#  start matlab engine
eng = matlab.engine.start_matlab()

for data_fname in data_fnames:
        for sampling in samplings:
            folder_out = '{}/{}'.format(output_folder, data_fname) 
            folder_out += '_sampling_{}'.format(sampling)
            if deformation is 'rigid':
                folder_out += '_shift_{}-{}_rot_{}'.format(shift[0],shift[1],phi)
            elif deformation is 'zoom':
                folder_out += '_zoom_{}'.format(zoomfactor)
            elif deformation is 'zoom_rigid':
#                    folder_out += '_shift_{}-{}_rot_{}'.format(shift[0],shift[1],phi)
                folder_out += '_zoomr_{}'.format(zoomfactor)    
            else:
                folder_out += '_mixed'
                
            folder_out += '_noise_{}'.format(stddev) 
            folder_out += '_NonNeg_{}'.format(NonNeg)
            folder_out += '_MS_coord'
            
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
                
            # Target space
            gt, sinfo = np.load('{}/{}.npy'.format(data_folder, data_fname))
            simage = sinfo.shape
            complex_space = odl.uniform_discr([-1, -1], [1, 1], simage, 
                                              dtype='complex', interp='linear')
            X = complex_space.real_space ** 2
            V = X[0].tangent_bundle
            Yaff = odl.tensor_space(6)
            Z = odl.ProductSpace(X, Yaff) 
            
            J = ops.Complex2Real(complex_space)
            M = ops.Magnitude(X)
            
            # Ground truth image
            phase = complex_space.element(
                    lambda x: np.exp(1j * .8 * (np.sin(3 * x[0] ** 2) + 
                                                np.cos(5 * x[1] ** 3))))
            
            gt = J(complex_space.element(gt * phase))
            sinfo = complex_space.real_space.element(sinfo)
            
            mask_bdr = np.zeros(sinfo.shape)
            mask_bdr[25:-24,25:-24] = 1

            gt_bdr = gt*mask_bdr
            gt = gt_bdr
            sinfo_bdr = sinfo*mask_bdr
            sinfo = sinfo_bdr
        
            
            # Generate affine map and data image
            if deformation is 'rigid':
                cosp = np.cos(phi)
                sinp = np.sin(phi)
                disp_func = [
                        lambda x: (cosp-1)*x[0] - sinp*x[1] + shift[0],
                        lambda x: sinp*x[0] + (cosp-1)*x[1] + shift[1]]
                vf_gt = Yaff.element([shift[0], shift[1], 
                                      cosp-1,-sinp, 
                                      sinp, cosp-1])
            elif deformation is 'zoom':
                disp_func = [
                        lambda x: (zoomfactor-1)*x[0] ,
                        lambda x: (zoomfactor-1)*x[1] ]
                vf_gt = Yaff.element([0, 0, 
                                      zoomfactor-1, 0, 
                                      0,zoomfactor-1])
            elif deformation is 'mixed':
                sinp = np.sin(phi)
                disp_func = [
                        lambda x: (zoomfactor-1)*x[0] + 0.04*x[1] + shift[0],
                        lambda x: sinp*x[0] + (zoomfactor-1)*x[1] + shift[1]]
                vf_gt = Yaff.element([shift[0], shift[1], 
                                      zoomfactor-1, 0.04, 
                                      sinp, zoomfactor-1])
            elif deformation is 'zoom_rigid':
                cosp = np.cos(phi)
                sinp = np.sin(phi)
                disp_func = [
                        lambda x: (zoomfactor*cosp-1)*x[0] - zoomfactor*sinp*x[1] + shift[0],
                        lambda x: zoomfactor*sinp*x[0] + (zoomfactor*cosp-1)*x[1] + shift[1]]
                vf_gt = Yaff.element([shift[0], shift[1], 
                                      zoomfactor*cosp-1, -zoomfactor*sinp, 
                                      zoomfactor*sinp, zoomfactor*cosp-1])

                                            
            
            vf = V.element(disp_func)
            
            X_plot = odl.uniform_discr([-1, -1], [1, 1], (15, 15), interp='linear')
            V_plot = X_plot.tangent_bundle
            
            x,y = X_plot.meshgrid
            u = disp_func[0]([x,y])         
            v = disp_func[1]([x,y]) 
            
            plt.quiver(x,y,u,v,color='yellow')
            plt.axis('equal'),
            plt.axis([-1.3,1.3,-1.3,1.3]),
#                plt.axis('off'),
            plt.gca().set_aspect('equal', adjustable='box'),
            plt.savefig(folder_out + '/deformation_field.png',
                        bbox_inches='tight', 
                        pad_inches=0, 
                        transparent=True, dpi=600,format='png')

                            
            deform_op = defm.LinDeformFixedDisp(vf)
                    
            deformed_gt = odl.DiagonalOperator(deform_op, deform_op)(gt)
            
            # Forward operator    
            if unitary is True:
                F = ops.UnitaryRealFourierTransform(X)
            else:
                F = ops.RealFourierTransform(X)
            
            sampling_parts = sampling.split('-')
            if sampling_parts[0] == 'cartesian':
                r = int(sampling_parts[1])
                sampling_op, mask = ops.get_cartesian_sampling(F.range[0], r)
            elif sampling_parts[0] == 'random':
                p = float(sampling_parts[1])
                sampling_op, mask = ops.get_random_sampling(F.range[0], p)
            elif sampling_parts[0] == 'radial':
                num_angles = int(sampling_parts[1])
                sampling_op, mask = ops.get_radial_sampling(F.range[0], num_angles)                
            elif sampling_parts[0] == 'radialbl':
                num_angles = int(sampling_parts[1])
                sampling_op, mask = ops.get_radial_sampling(F.range[0], num_angles, block=10)                       
            
            S = odl.DiagonalOperator(sampling_op, 2)  
            
            
            A = S * F
            
            
            # data
            noise = odl.phantom.white_noise(A.range, mean=0, stddev=stddev, seed=1)
            aligned_data = A(gt) + noise
            deformed_data = A(deformed_gt) + noise
            
            # Save data images
            plt.imsave(folder_out + '/' + data_fname + '_sampling_pattern.png', 
                       mask, cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_data.png', 
                       np.fft.fftshift(M(F(gt))), cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_data_deformed.png', 
                       np.fft.fftshift(M(F(deformed_gt))), cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_ift_deformed.png', 
                       M(F.inverse(S.adjoint(deformed_data))), cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_ift.png', 
                       M(F.inverse(S.adjoint(aligned_data))), cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_gt_deformed.png', 
                       M(deformed_gt), cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_gt.png', 
                       M(gt), cmap='gray')
            plt.imsave(folder_out + '/' + data_fname + '_sinfo.png', 
                       sinfo, cmap='gray')
            
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
                datafit = 0.5 * odl.solvers.functional.L2NormSquared(aligned_data.space).translated(aligned_data)
            else:
                datafit = 0.5 * odl.solvers.functional.L2NormSquared(deformed_data.space).translated(deformed_data)

            f = fctls.DataFitDisp(Z, datafit, forward=A)         
            
            
            for it in range(2):
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
                                    
                g = odl.solvers.SeparableSum(reg_im, reg_affine) 
                obj = f + g
                
                x_init = Z.element([im_init,vf_init])
             
                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True) &
                      odl.solvers.CallbackPrint(step=step, func=obj, fmt='obj={!r}') & 
                      odl.solvers.CallbackShow(step=step))
                
                ud_vars = [0]
                file_out = '{}_{}_da{}_si{}_vars{}_a{:4.2e}_g{:6.4f}_e{:6.4f}'.format('recon', 
                        it, data_aligned, use_sinfo, ud_vars, RegParam, gamma, eta)
                 
            
                recon = algs.PALM(f, g, ud_vars=ud_vars.copy(), x=x_init, niter=None, 
                                     callback=cb, L=None, tol=1e-9)    
            
            
                niter_diff = [niter[0]] + list(np.diff(niter)) 
              
                clim = [0, 1] 
                proj = odl.solvers.IndicatorBox(M.range, lower=clim[0], upper=clim[1]).proximal(1) * M
                
                for ni, nid in zip(niter, niter_diff):
                    recon.run(nid)   
                    if save2Disk is True:    
                        file_out_ni = '{}_{:04d}'.format(file_out, ni) 
                
                        plt.imsave(folder_out + '/' + file_out_ni + '.png', 
                                   proj(recon.x[0]), cmap='gray')
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

                
            ssim_val = ssim(mask_bdr*M(recon.x[0]),M(gt))
            file1 = open("{}.txt".format(folder_out + '/' + file_out_ni),"w")   
            file1.write("SSIM: {} \n".format(ssim_val)) 
            file1.write("RD Vf: {} \n".format(rel_diff_vf)) 
            file1.close()
            
eng.quit()

            
             