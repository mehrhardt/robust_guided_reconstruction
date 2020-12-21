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
stddev      = 0.001                     # standard deviation for gaussian noise
phi         = 0.1
shift       = [-0.02, -0.08]         
zoomfactor  = 0.85
deformation = 'zoom_rigid'              #'zoom', 'rigid', 'mixed'
NonNeg      = False
unitary     = False

# Algorithmic parameters
save2Disk   = True
niter       = [10, 20, 50, 100, 500]    # number of PALM iterations per resolution
resolutions = [32,64,128,256]       
data_aligned = False

# Vectorfield parameters
gamma = 0.9995
eta = 1e-2

# Define reconstruction settings
settings = ['tv_recon', 'dtv_affreg_recon']

data_folder = '../data'
data_fnames = ['mri_patienta_ch0']  #['mri_brainweba_ch0', 'mri_phantom_ch0']
samplings =  ['radialbl-15']        #['cartesian-2', 'cartesian-4','random-0.5','radial-70','radialbl-30',]


output_folder = '../results'

for data_fname in data_fnames:  
    for sampling in samplings:
        for setting in settings:
            if setting == 'tv_recon':
                use_sinfo   = False
                ud_vars     = [0]
                alpha       = 5e-4
            elif setting == 'dtv_affreg_recon':
                use_sinfo   = True
                ud_vars     = [0,1]
                alpha       = 5e-3
            else:
                raise ValueError('Unknown setting')
            
            
            folder_out = '{}/{}'.format(output_folder, data_fname) 
            folder_out += '_sampling_{}'.format(sampling)
            if deformation == 'rigid':
                folder_out += '_shift_{}-{}_rot_{}'.format(shift[0],shift[1],phi)
            elif deformation == 'zoom':
                folder_out += '_zoom_{}'.format(zoomfactor)
            elif deformation == 'zoom_rigid':
                folder_out += '_zoomr_{}'.format(zoomfactor)    
            else:
                folder_out += '_mixed'
                
            folder_out += '_noise_{}'.format(stddev) 
            folder_out += '_multigrid'
            
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
            
            gt = J(complex_space.element(sktransform.resize(gt, simage)) * phase)
            sinfo = complex_space.real_space.element(sktransform.resize(sinfo, simage))
            
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
                                      cosp-1, -sinp, 
                                      sinp, cosp-1])
            elif deformation is 'zoom':
                disp_func = [
                        lambda x: (zoomfactor-1)*x[0] ,
                        lambda x: (zoomfactor-1)*x[1] ]
                vf_gt = Yaff.element([0, 0, 
                                      zoomfactor*-1, 0, 
                                      0, zoomfactor-1])
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
            
            file1 = open("{}.txt".format(folder_out + '/' + 'PsInv'),"w")   
            ssim_val_psinv = ssim(M(F.inverse(S.adjoint(aligned_data))),M(gt))
            file1.write("SSIM: {} \n".format(ssim_val_psinv))
            file1.close()
            
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
            complex_space_first = odl.uniform_discr([-1, -1], [1, 1], 
                                                    (resolutions[0],resolutions[0]),
                                                    dtype='complex', interp='linear')
            X_first = complex_space_first.real_space ** 2
            im_init = X_first.zero()
            vf_init = Yaff.zero()
            
            ind = 0
            alphas = [alpha*5**i for i in range(len(resolutions))]
            alphas.reverse()
            
                
            for res in resolutions:
                RegParam = alphas[ind]
                
                simage_res = (res,res)
                if use_sinfo is True:
                    sinfo_res = sktransform.resize(sinfo, simage_res)
                else:
                    sinfo_res = None
                        
                
                complex_space_res = odl.uniform_discr([-1, -1], [1, 1], 
                                                    simage_res, dtype='complex', 
                                                    interp='linear')
                X_res = complex_space_res.real_space ** 2
                V_res = X_res[0].tangent_bundle
                Z_res = odl.ProductSpace(X_res, Yaff) 
                
                # Forward operator                         
                upsampling = odl.DiagonalOperator(ops.Subsampling(X[0], X_res[0], 0).adjoint, 2)                              
                A_res = S * F * upsampling    
                
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

                f = fctls.DataFitDisp(Z_res, datafit, forward=A_res)                              
                
                
                file_out = 'recon_da{}_si{}_vars{}_a{:4.2e}_g{:6.4f}_e{:6.4f}_res{}'.format(
                         data_aligned, use_sinfo, ud_vars, RegParam, gamma, eta, res)
                                
                reg_im = fctls.TV(X_res, alpha=RegParam, sinfo=sinfo_res, 
                                  NonNeg=NonNeg, 
                                  prox_options=prox_options.copy(), 
                                  gamma=gamma, eta=eta)
                g = odl.solvers.SeparableSum(reg_im, reg_affine) 
            
                # Define objective functional
                obj = f + g
                
                x_init = Z_res.element([im_init,vf_init])
             
                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True) &
                      odl.solvers.CallbackPrint(step=step, func=obj, fmt='obj={!r}') & 
                      odl.solvers.CallbackShow(step=step))
            
                recon = algs.PALM(f, g, ud_vars=ud_vars.copy(), x=x_init, niter=None, 
                                     callback=cb, L=None, tol=1e-9)    
            
            
                niter_diff = [niter[0]] + list(np.diff(niter)) 
              
                M_res = ops.Magnitude(X_res)
                clim = [0, 1] 
                proj = odl.solvers.IndicatorBox(M_res.range, lower=clim[0], upper=clim[1]).proximal(1) * M_res
                
                for ni, nid in zip(niter, niter_diff):
                    recon.run(nid)   
                      
                    if 1 in ud_vars:
                        vf =  ops.Embedding_Affine(Yaff, V_plot)(recon.x[1])
                        plt.quiver(x, y, vf[0], vf[1], color='red'),
                        plt.axis('equal'),
                        plt.axis([-1.3,1.3,-1.3,1.3]),
                        plt.gca().set_aspect('equal', adjustable='box'),
                    if save2Disk is True:    
                        file_out_ni = '{}_{:04d}'.format(file_out, ni) 
                        if 1 in ud_vars:
                            file_out_def_ni = '{}_def_{:04d}'.format(file_out, ni) 
                            plt.savefig(folder_out + '/' + file_out_def_ni + '.png', 
                                        bbox_inches='tight', 
                                        pad_inches=0, 
                                        transparent=True, 
                                        dpi=600, format='png')
                        plt.imsave(folder_out + '/' + file_out_ni + '.png', 
                                   proj(recon.x[0]), cmap='gray')
                        np.save('{}.npy'.format(folder_out + '/' + file_out_ni), list(recon.x))
                
                
                rel_diff_vf = (recon.x[1]-vf_gt).norm()/vf_gt.norm()
                file1 = open("{}.txt".format(folder_out + '/' + file_out_ni + '_def'),"w")   
                file1.write("RelDiff VF: {} \n".format(rel_diff_vf)) 
                file1.close()
                
                complex_space_new = odl.uniform_discr([-1, -1], [1, 1], 
                                                    (2*res,2*res), dtype='complex', 
                                                    interp='linear')
                X_new = complex_space_new.real_space ** 2
            
                sampling = odl.DiagonalOperator(ops.Subsampling(X_new[0], X_res[0], 0), 2)
                im_init = sampling.adjoint(recon.x[0]).copy()
                vf_init = recon.x[1].copy()
                ind += 1
            
            ssim_val = ssim(mask_bdr*M(recon.x[0]),M(gt))
            rel_diff_vf = (recon.x[1]-vf_gt).norm()/vf_gt.norm()
            
            file1 = open("{}.txt".format(folder_out + '/' + file_out_ni),"w")   
            file1.write("SSIM: {} \n".format(ssim_val)) 
            if use_sinfo is False:
                ssim_val_def = ssim(M(recon.x[0]),M(deformed_gt))
                file1.write("SSIM def: {} \n".format(ssim_val_def))
            file1.write("RelDiff VF: {} \n".format(rel_diff_vf)) 
            file1.close()
                
                