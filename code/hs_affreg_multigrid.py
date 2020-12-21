import numpy as np
import odl
import functionals as fctls
import algorithms as algs
import operators as ops
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import deform as defm
import os

#%%
# Model parameters
phi     = 0.1                       # rotation between side info and data
shift   = [0.06 , -0.04]            # shift between side info and data
deformations = ['rigid','shear','nonlinear']      # type of deformation
colormap = 'viridis'                # 'gnist_ncar', 'viridis', 'nipy_spectral'

# Algorithmic parameters
save2Disk   = True
niter       = [10, 20, 50, 100, 200, 500]   # number of PALM iterations per resolution
factors     = [0.25,0.5,1,2,4] 
margin      = 10                            # for data creation
data_aligned = False
use_sinfo   = True
ud_vars     = [0,1]

# Vectorfield parameters 
gamma = 0.9995
eta  = 1e-2    

# Define reconstruction settings
settings = ['tv_recon', 'dtv_noreg_recon','target_recon','dtv_affreg_recon']

data_folder = '../data'
data_fnames = ['urban_ch3_0-100x100-200']


output_folder = '../results'

for data_fname in data_fnames:
    for deformation in deformations:
        if deformation != 'rigid':
            settings = [settings[-1]]
            
        for setting in settings:
            if setting == 'tv_recon':
                use_sinfo   = False
                ud_vars     = [0]
                alpha       = 1e-5
            elif setting == 'dtv_noreg_recon':
                use_sinfo   = True
                ud_vars     = [0]
                alpha       = 1e-3
            elif setting == 'target_recon':
                use_sinfo   = True
                ud_vars     = [0,1]
                alpha       = 1e-3
                data_aligned = True
            elif setting == 'dtv_affreg_recon':
                use_sinfo   = True
                ud_vars     = [0,1]
                alpha       = 1e-3
            else:
                raise ValueError('Unknown setting: {}'.format(setting))
    
            folder_out = '{}/{}'.format(output_folder, data_fname) 
            
            if deformation == 'rigid':
                folder_out += '_shift_{}-{}_rot_{}'.format(shift[0],shift[1],phi) 
            else:
                folder_out += '_{}'.format(deformation)
                   
            folder_out += '_multigrid'
            
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)
                
            # Spaces
            data, sinfo = np.load('{}/{}.npy'.format(data_folder, data_fname), 
                                  allow_pickle=True)
            
            sdata = data.shape
            Y = odl.uniform_discr([-1, -1], [1, 1], sdata, interp='linear')
            V = Y.tangent_bundle
            Yaff = odl.tensor_space(6)
            Xside = odl.uniform_discr([-1, -1], [1, 1], sinfo.shape, interp='linear')
            sinfo = Xside.element(sinfo)
            
            # create data
            # Data image
            aligned_data = Y.element(data)
            #create margin
            aligned_data[0:margin,:] = 0
            aligned_data[:,0:margin] = 0
            aligned_data[-(margin+1):-1,:] = 0
            aligned_data[-1,:] = 0
            aligned_data[:,-(margin+1):-1] = 0
            aligned_data[:,-1] = 0
        
            # Generate affine map and data image
            if deformation == 'rigid':
                cosp = np.cos(phi)
                sinp = np.sin(phi)
                disp_func = [
                        lambda x: (cosp-1)*x[0] - sinp*x[1] + shift[0],
                        lambda x: sinp*x[0] + (cosp-1)*x[1] + shift[1]] 
                vf_gt = Yaff.element([shift[0], shift[1], 
                                          cosp-1, -sinp, 
                                          sinp, cosp-1])
            elif deformation == 'shear':
                disp_func = [
                        lambda x: 0*x[0] + 0.08*x[1] + shift[0],
                        lambda x: 0*x[0] + 0*x[1] + shift[1]]
                vf_gt = Yaff.element([shift[0], shift[1], 
                                          0, 0.08, 
                                          0, 0])
            elif deformation == 'nonlinear':  
                cosp = np.cos(phi)
                sinp = np.sin(phi)
                disp_func = [
                        lambda x: (cosp-1)*x[0] - sinp*x[1] + 0.05*x[1]**2 + shift[0],
                        lambda x: sinp*x[0] + (cosp-1)*x[1] - 0.05*x[0]**3 + shift[1]]
                vf_gt = Yaff.element([shift[0], shift[1], 
                                          cosp-1, -sinp, 
                                          sinp, cosp-1])
                
        
            vf = V.element(disp_func)
            vf_hr = Xside.tangent_bundle.element(disp_func)
        
            deform_op = defm.LinDeformFixedDisp(vf)
            deform_op_hr = defm.LinDeformFixedDisp(vf_hr)
           
            # create clipping operator and deformed data image
            clim = [0, np.max(aligned_data)] 
            if colormap == 'viridis':
                clim[1] *= 0.7
            
            projY = odl.solvers.IndicatorBox(Y, lower=clim[0], upper=clim[1]).proximal(1)    
        
            deformed_data = projY(deform_op(aligned_data))
            
            plt.imsave(folder_out + '/' + data_fname + '_data_deformed.png', 
                       projY(deformed_data), cmap=colormap)
            plt.imsave(folder_out + '/' + data_fname + '_data.png', 
                       projY(aligned_data), cmap=colormap)
            plt.imsave(folder_out + '/' + data_fname + '_sinfo.png', sinfo, cmap='gray')
                 
            
            ind = 0
            alphas = [alpha*10**i for i in range(len(factors))]
            alphas.reverse()
            for factor in factors:
                regParam = alphas[ind]
                simage = tuple(int(factor*x) for x in sdata)
                X = odl.uniform_discr([-1, -1], [1, 1], simage, interp='linear')
                
                if use_sinfo is True:
                    sinfo_im = ops.Subsampling(Xside ,X, 0)(sinfo)
                else:
                    sinfo_im = None
            
                
                #%%
                Z = odl.ProductSpace(X, Yaff)         
                
                if factor < 1:
                    A = ops.Subsampling(Y, X, 0).adjoint
                else:
                    A = ops.Subsampling(X, Y, 0)
                    
                projX = odl.solvers.IndicatorBox(X, lower=clim[0], upper=clim[1]).proximal(1)    
                
                
                
                # Set some parameters and the general TV prox options
                prox_options = {}
                prox_options['name'] = 'FGP'
                prox_options['warmstart'] = True
                prox_options['p'] = None
                prox_options['tol'] = None
                prox_options['niter'] = 10  
                strong_convexity = 0
                              
                step = 10
                
                if data_aligned is True:
                    datafit = 0.5 * odl.solvers.functional.L2NormSquared(aligned_data.space).translated(aligned_data)
                else:
                    datafit = 0.5 * odl.solvers.functional.L2NormSquared(deformed_data.space).translated(deformed_data)

                f = fctls.DataFitDisp(Z, datafit, forward=A)
    
                
                reg_im = fctls.TV(X, alpha=regParam, sinfo=sinfo_im, NonNeg=True, 
                                  prox_options=prox_options.copy(), 
                                  gamma=gamma, eta=eta)
                    
                reg_affine = odl.solvers.ZeroFunctional(Yaff)
         
        
                g = odl.solvers.SeparableSum(reg_im, reg_affine)
                    
                # Define objective functional
                obj = f + g
                 
                cb = (odl.solvers.CallbackPrintIteration(end=', ') &
                      odl.solvers.CallbackPrintTiming(cumulative=False, end=', ') &
                      odl.solvers.CallbackPrintTiming(fmt='total={:.3f} s', cumulative=True) &
                      odl.solvers.CallbackPrint(step=step, func=obj, fmt='obj={!r}') & 
                      odl.solvers.CallbackShow(step=step))
                  
                if ind == 0:
                    im_init = X.zero()
                    vf_init = Yaff.zero()
                else:
                    upsampling_fac = int(factors[ind]/factors[ind-1])
                    im_init = np.kron(recon.x[0], np.ones((upsampling_fac, upsampling_fac)))
                    vf_init = recon.x[1].copy()
                
                x_init = Z.element([im_init,vf_init])
            
                recon = algs.PALM(f, g, ud_vars=ud_vars.copy(), x=x_init, niter=None, 
                                 callback=cb, L=None, tol=1e-9)    
                    
                
                niter_diff = [niter[0]] + list(np.diff(niter)) 
                
                file_out = '{}_da{}_si{}_vars{}_a{:4.2e}_g{:6.4f}_e{:6.4f}_factor{}'.format(data_fname, 
                         data_aligned, use_sinfo, ud_vars, regParam, gamma, eta, factor)
                
        
                for ni, nid in zip(niter, niter_diff):
                    recon.run(nid)
                    if save2Disk is True:
                        file_out_ni = '{}_{:04d}'.format(file_out, ni)
                        plt.imsave(folder_out + '/' + file_out_ni + '.png', 
                                   projX(recon.x[0]), cmap=colormap)
                        np.save('{}.npy'.format(folder_out + '/' + file_out_ni), list(recon.x))
                
                if save2Disk is True:
                    rel_diff_vf = (recon.x[1]-vf_gt).norm()/vf_gt.norm()
                    file1 = open("{}.txt".format(folder_out + '/' + file_out_ni + '_def'),"w")   
                    file1.write("RelDiff VF: {} \n".format(rel_diff_vf)) 
                    file1.close()
                
                ind += 1
                
            rel_diff_vf = (recon.x[1]-vf_gt).norm()/vf_gt.norm()
            
            file1 = open("{}.txt".format(folder_out + '/' + file_out_ni),"w")   
            file1.write("RelDiff VF: {} \n".format(rel_diff_vf)) 
            file1.close()    