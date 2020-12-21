import numpy as np
import odl
import operators as ops
from odl.solvers.functional.functional import Functional
from odl.solvers import (GroupL1Norm, ZeroFunctional, IndicatorBox,
                         L2NormSquared, L1Norm, Huber)
from odl.space.pspace import ProductSpace
from odl.operator.pspace_ops import BroadcastOperator
from odl.operator.operator import Operator
from odl.operator.default_ops import (IdentityOperator, 
                                      ConstantOperator, ZeroOperator)
from odl.operator.tensor_ops import PointwiseInner
import deform as defm

from algorithms import fgp_dual

##################################################
#################HELPER FUNCTIONS#################
##################################################


def total_variation(domain, grad): 
    L1 = GroupL1Norm(grad.range, exponent=2)
    return L1 * grad


def generate_vfield_from_sinfo(sinfo, grad, eta=1e-2):
    sinfo_grad = grad(sinfo)
    grad_space = grad.range
    norm = odl.PointwiseNorm(grad_space, 2)
    norm_sinfo_grad = norm(sinfo_grad)
    max_norm = np.max(norm_sinfo_grad)
    eta_scaled = eta * max(max_norm, 1e-4)
    norm_eta_sinfo_grad = np.sqrt(norm_sinfo_grad ** 2 +
                                  eta_scaled ** 2)
    xi = grad_space.element([g / norm_eta_sinfo_grad for g in sinfo_grad])

    return xi

    
def project_on_fixed_vfield(domain, vfield):

        class OrthProj(Operator):
            def __init__(self):
                super(OrthProj, self).__init__(domain, domain, linear=True)

            def _call(self, x, out):
                xi = vfield
                Id = IdentityOperator(domain)
                xiT = odl.PointwiseInner(domain, xi)
                xixiT = odl.BroadcastOperator(*[x*xiT for x in xi])
                gamma = 1
                P = (Id - gamma * xixiT)
                out.assign(P(x))

            @property
            def adjoint(self):
                return self

            @property
            def norm(self):
                return 1.

        return OrthProj()
    
    
#############################################
#################FUNCTIONALS#################
#############################################
    

class TV(Functional):

    def __init__(self, domain, alpha=1., sinfo=None, NonNeg=False,
                 prox_options={}, gamma=1, eta=1e-2):
                 
        if isinstance(domain, odl.ProductSpace):
            grad_basic = odl.Gradient(
                    domain[0], method='forward', pad_mode='symmetric')
            
            pd = [odl.discr.diff_ops.PartialDerivative(
                    domain[0], i, method='forward', pad_mode='symmetric') 
                  for i in range(2)]
            cp = [odl.operator.ComponentProjection(domain, i) 
                  for i in range(2)]
                
            if sinfo is None:
                self.grad = odl.BroadcastOperator(
                        *[pd[i] * cp[j]
                          for i in range(2) for j in range(2)])
                
            else:
                vfield = gamma * generate_vfield_from_sinfo(sinfo, grad_basic, eta)
                inner = odl.PointwiseInner(domain, vfield) * grad_basic
                self.grad = odl.BroadcastOperator(
                        *[pd[i] * cp[j] - vfield[i] * inner * cp[j] 
                          for i in range(2) for j in range(2)])
            
            self.grad.norm = self.grad.norm(estimate=True)
            
        else:
            grad_basic = odl.Gradient(
                    domain, method='forward', pad_mode='symmetric')
            
            if sinfo is None:
                self.grad = grad_basic
            else:
                vfield = gamma * generate_vfield_from_sinfo(sinfo, grad_basic, eta)
                P = project_on_fixed_vfield(grad_basic.range, vfield)
                self.grad = P * grad_basic
                
            grad_norm = 2 * np.sqrt(sum(1 / grad_basic.domain.cell_sides**2))
            self.grad.norm = grad_norm
        
        self.tv = total_variation(domain, grad=self.grad)

        if NonNeg is True:
            self.nn = IndicatorBox(domain, 0, np.inf)
        else:
            self.nn = ZeroFunctional(domain)            

        if 'name' not in prox_options:
            prox_options['name'] = 'FGP'
        if 'warmstart' not in prox_options:
            prox_options['warmstart'] = True
        if 'niter' not in prox_options:
            prox_options['niter'] = 5
        if 'p' not in prox_options:
            prox_options['p'] = None
        if 'tol' not in prox_options:
            prox_options['tol'] = None

        prox_options['grad'] = self.grad
        prox_options['proj_P'] = self.tv.left.convex_conj.proximal(0)
        prox_options['proj_C'] = self.nn.proximal(1)

        self.prox_options = prox_options
        self.alpha = alpha

        super().__init__(space=domain, linear=False, grad_lipschitz=0)

    def __call__(self, x):
        if self.alpha == 0:
            return 0
        else:
            nn = self.nn(x)
            out = self.alpha * self.tv(x) + nn
            return out

    @property
    def proximal(self):
        alpha = self.alpha
        prox_options = self.prox_options
        space = self.domain
        
        class ProximalTV(Operator):
            def __init__(self, sigma):
                self.sigma = float(sigma)
                self.prox_options = prox_options
                self.alpha = float(alpha)
                super(ProximalTV, self).__init__(
                    domain=space, range=space, linear=False)
    
            def _call(self, z, out):
                sigma = self.sigma * self.alpha
                if sigma == 0:
                    out.assign(z)
                else:
                    opts = self.prox_options
    
                    grad = opts['grad']
                    proj_C = opts['proj_C']
                    proj_P = opts['proj_P']
    
                    if opts['name'] == 'FGP':
                        if opts['warmstart']:
                            if opts['p'] is None:
                                opts['p'] = grad.range.zero()
    
                            p = opts['p']
                        else:
                            p = grad.range.zero()
    
                        niter = opts['niter']
                        out.assign(fgp_dual(p, z, sigma, niter, grad, proj_C,
                                          proj_P, tol=opts['tol']))
    
                    else:
                        raise NotImplementedError('Not yet implemented')
                    
        return ProximalTV

   
class DataFitL2Disp(Functional):

    def __init__(self, space, data, forward=None):
        self.space = space
        self.image_space = self.space[0]
        self.affine_space = self.space[1]
        self.data = data
        if forward is None:
            self.forward = IdentityOperator(self.image_space)
        else:
            self.forward = forward
        
        self.datafit = 0.5 * L2NormSquared(data.space).translated(self.data)
        
        if isinstance(self.image_space, odl.ProductSpace):
            tangent_bundle = self.image_space[0].tangent_bundle
        else:
            tangent_bundle = self.image_space.tangent_bundle

        self.embedding = ops.Embedding_Affine(
                self.affine_space, tangent_bundle)
            
        super(DataFitL2Disp, self).__init__(space=space, linear=False,
                                            grad_lipschitz=np.nan)
    
    def __call__(self, x):
        xim = x[0]
        xaff = x[1]
        transl_operator = self.transl_op_fixed_im(xim)
        fctl = self.datafit * self.forward * transl_operator
        return fctl(xaff)
    
    
    def transl_op_fixed_im(self, im):
        
        if isinstance(self.image_space, odl.ProductSpace):
            deform_op = odl.BroadcastOperator(
                    defm.LinDeformFixedTempl(im[0]),
                    defm.LinDeformFixedTempl(im[1]))
        else:
            deform_op = defm.LinDeformFixedTempl(im)
            
        return deform_op * self.embedding
    
    
    def transl_op_fixed_vf(self, disp):
                
        deform_op = defm.LinDeformFixedDispAffine(self.embedding(disp), disp)
        
        if isinstance(self.image_space, odl.ProductSpace):
            deform_op = odl.DiagonalOperator(deform_op, len(self.image_space))

        return deform_op   
    
    
    def partial_gradient(self, i):
        if i == 0:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.image_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]
                    transl_operator = functional.transl_op_fixed_vf(xaff)                  
                    func0 = functional.datafit * functional.forward * transl_operator
                    grad0 = func0.gradient                    
                    out.assign(grad0(xim))
                    
            return auxOperator()
        elif i == 1:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.affine_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]                    
                    transl_operator = functional.transl_op_fixed_im(xim)                  
                    func1 = functional.datafit * functional.forward * transl_operator
                    grad1 = func1.gradient    
                    out.assign(grad1(xaff))                            
            return auxOperator()
        else:
            raise ValueError('No gradient defined for this variable')

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2)])
                

class DataFitDisp(Functional):

    def __init__(self, space, datafit, forward=None):
        self.space = space
        self.image_space = self.space[0]
        self.affine_space = self.space[1]

        if forward is None:
            self.forward = IdentityOperator(self.image_space)
        else:
            self.forward = forward
        
        self.datafit = datafit
        
        if isinstance(self.image_space, odl.ProductSpace):
            tangent_bundle = self.image_space[0].tangent_bundle
        else:
            tangent_bundle = self.image_space.tangent_bundle

        self.embedding = ops.Embedding_Affine(
                self.affine_space, tangent_bundle)
            
        super(DataFitDisp, self).__init__(space=space, linear=False,
                                            grad_lipschitz=np.nan)
    
    def __call__(self, x):
        xim = x[0]
        xaff = x[1]
        transl_operator = self.transl_op_fixed_im(xim)
        fctl = self.datafit * self.forward * transl_operator
        return fctl(xaff)
    
    
    def transl_op_fixed_im(self, im):
        
        if isinstance(self.image_space, odl.ProductSpace):
            deform_op = odl.BroadcastOperator(
                    defm.LinDeformFixedTempl(im[0]),
                    defm.LinDeformFixedTempl(im[1]))
        else:
            deform_op = defm.LinDeformFixedTempl(im)
            
        return deform_op * self.embedding
    
    
    def transl_op_fixed_vf(self, disp):
                
        # deform_op = defm.LinDeformFixedDisp(self.embedding(disp))
        deform_op = defm.LinDeformFixedDispAffine(self.embedding(disp), disp)
        
        if isinstance(self.image_space, odl.ProductSpace):
            deform_op = odl.DiagonalOperator(deform_op, len(self.image_space))

        return deform_op   
    
    
    def partial_gradient(self, i):
        if i == 0:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.image_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]
                    transl_operator = functional.transl_op_fixed_vf(xaff)                  
                    func0 = functional.datafit * functional.forward * transl_operator
                    grad0 = func0.gradient                    
                    out.assign(grad0(xim))
                    
            return auxOperator()
        elif i == 1:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.affine_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]                    
                    transl_operator = functional.transl_op_fixed_im(xim)                  
                    func1 = functional.datafit * functional.forward * transl_operator
                    grad1 = func1.gradient    
                    out.assign(grad1(xaff))                            
            return auxOperator()
        else:
            raise ValueError('No gradient defined for this variable')

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2)])                

class DataFitL2DispAffRest(Functional):

    def __init__(self, space, data, forward=None):
        self.space = space
        self.image_space = self.space[0]
        self.affine_space = self.space[1]
        self.rest_space = self.space[2]
        self.deformation_space = ProductSpace(self.affine_space,self.rest_space)
        self.data = data
        if forward is None:
            self.forward = IdentityOperator(self.image_space)
        else:
            self.forward = forward
        
        self.datafit = 0.5 * L2NormSquared(self.image_space).translated(self.data)
        self.embedding_affine_rest = ops.Embedding_Affine_Rest(
                self.deformation_space, self.image_space.tangent_bundle)
        self.embedding_affine = ops.Embedding_Affine(
                self.affine_space, self.image_space.tangent_bundle)
            
        super(DataFitL2DispAffRest, self).__init__(space=space, linear=False,
                                               grad_lipschitz=np.nan)
    
    def __call__(self, x):
        xim = x[0]
        xaff = x[1]
        xrest = x[2]
        xdeform = self.deformation_space.element([xaff, xrest])
        transl_operator = self.transl_op_fixed_vf(xdeform)
        fctl = self.datafit * self.forward * transl_operator
        return fctl(xim)
    
    
    def transl_op_fixed_im_aff(self, im, aff):
        affine_deform = defm.LinDeformFixedDisp(self.embedding_affine(aff))
        deform_op = defm.LinDeformFixedTempl(affine_deform(im))
        transl_operator = deform_op 
        return transl_operator
    
    def transl_op_fixed_im_rest(self, im, rest):
        rest_deform = defm.LinDeformFixedDisp(rest)
        deformed_im = rest_deform(im)
        transl_operator = defm.LinDeformFixedTempl(deformed_im) * self.embedding_affine
        return transl_operator
    
    def transl_op_fixed_vf(self, disp):
        deform_op = defm.LinDeformFixedDisp(self.embedding_affine_rest(disp))
        transl_operator = deform_op
        return transl_operator   
    
    def partial_gradient(self, i):
        if i == 0:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.image_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]
                    xrest = x[2]
                    xdeform = functional.deformation_space.element([xaff, xrest])
                    transl_operator = functional.transl_op_fixed_vf(xdeform)                  
                    func = functional.datafit * functional.forward * transl_operator
                    grad = func.gradient                    
                    out.assign(grad(xim))                 
            return auxOperator()
        elif i == 1:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.affine_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]                    
                    xrest = x[2]
                    transl_operator = functional.transl_op_fixed_im_rest(xim,xrest)                  
                    func = functional.datafit * functional.forward * transl_operator
                    grad = func.gradient    
                    out.assign(grad(xaff))                 
            return auxOperator()
        elif i == 2:
            functional = self
            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.rest_space)
                def _call(self, x, out):
                    xim = x[0]
                    xaff = x[1]                    
                    xrest = x[2]
                    transl_operator = functional.transl_op_fixed_im_aff(xim,xaff)                  
                    func = functional.datafit * functional.forward * transl_operator
                    grad = func.gradient    
                    out.assign(grad(xrest))                 
            return auxOperator()
        else:
            raise ValueError('No gradient defined for this variable')

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(3)])

class DataFitL2TimeDep(Functional):

    def __init__(self, space, data, forward, alpha=None):
        
        self.N = round(len(space)/2)         # number of time steps
        self.space = space
        self.forward = forward
        self.data = data
        self.image_space = self.space[0]
        self.vf_space = self.space[self.N]
        self.space = space
        
        if alpha is None:
            alpha = np.ones(self.N)
        else:
            alpha = np.asarray(alpha)
        self.alpha = alpha
        super(DataFitL2TimeDep, self).__init__(space=space, linear=False,
                                               grad_lipschitz=np.nan)

    def __call__(self, x):
        ret = 0
        xim = x[:self.N]
        for i in range(self.N):
            if self.alpha[i] != 0:                
                xi = xim[i]
                res = self.residual(xi, i)
                ret += 0.5 * self.alpha[i] * L2NormSquared(res.space)(res)
        return ret

    def residual(self, x, i):
        return self.forward(x)-self.data[i]

    def spatial_image_gradient(self, i):
        if self.alpha[i] == 0:
            return ZeroOperator(self.space, range=self.image_space)
        else:
            functional = self

            class auxOperator(Operator):
                def __init__(self):
                    super(auxOperator, self).__init__(functional.space,
                                                      functional.image_space)

                def _call(self, x, out):
                    xim = x[:functional.N]
                    xi = xim[i]
                    ret = functional.alpha[i] * (
                            functional.forward.adjoint * (
                                    functional.forward -
                                    ConstantOperator(functional.data[i])))(xi)
                    out.assign(ret)

            return auxOperator()

    def spatial_vfield_gradient(self, i):
        i -= self.N
        return ZeroOperator(self.space, range=self.vf_space)

    def partial_gradient(self, i):
        if i < self.N:
            return self.spatial_image_gradient(i)
        else:
            return self.spatial_vfield_gradient(i)

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2*self.N)])


class L2OpticalFlowConstraint(Functional):

    def __init__(self, space, tau=1., alpha=1., grad=None):
        self.N = round(len(space)/2)         # number of time steps

        if grad is None:
            grad = odl.Gradient(space[0], method='forward',
                                pad_mode='symmetric')
            grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
        else:
            grad = grad

        self.grad = grad
        self.space = space
        self.image_space = self.space[0]
        self.vf_space = self.space[self.N]
        self.alpha = alpha
        self.tau = tau
        super(L2OpticalFlowConstraint,
              self).__init__(space=space, linear=False, grad_lipschitz=np.nan)

    def __call__(self, x):  # x is space time element im x vf
        ret = 0.
        xim = x[:self.N]
        xvf = x[self.N:]
        for i in range(self.N-1):
            ximi = xim[i]
            ximii = xim[i+1]
            xvfi = xvf[i]
            res = self.residual(ximii, ximi, xvfi)
            ret += 0.5 * L2NormSquared(res.space)(res)

        return self.alpha * ret

    
    def residual(self, im1, im2, vf):
        P = PointwiseInner(self.vf_space, vf)
        res = im1 - im2 + self.tau * P(self.grad(im1))

        return res


    def spatial_image_gradient(self, i):
        if self.alpha == 0:
            return ZeroOperator(self.space, range=self.image_space)
        else:
            functional = self
            if i == 0:
                class FirstImageDerivative(Operator):
                    def __init__(self):
                        super(FirstImageDerivative, self).__init__(
                                functional.space, functional.image_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        res = -functional.residual(xim[1], xim[0], xvf[0])
                        res *= functional.alpha
                        out.assign(res)

                return FirstImageDerivative()

            if i == self.N-1:
                class LastImageDerivative(Operator):
                    def __init__(self):
                        super(LastImageDerivative, self).__init__(
                                functional.space, functional.image_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        res = functional.residual(xim[-1], xim[-2], xvf[-2])
                        P = PointwiseInner(functional.vf_space, xvf[-2])
                        tmp = (res - functional.tau * 
                               (P(functional.grad(res)) - 
                                functional.grad.adjoint(xvf[-2]) * res))
                        tmp *= functional.alpha
                        out.assign(tmp)

                return LastImageDerivative()

            if 0 < i < self.N-1:
                class IntermediateImageDerivative(Operator):
                    def __init__(self):
                        super(IntermediateImageDerivative, self).__init__(
                                functional.space, functional.image_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        resi = functional.residual(xim[i], xim[i-1], xvf[i-1])
                        resii = functional.residual(xim[i+1], xim[i], xvf[i])
                        P = PointwiseInner(functional.vf_space, xvf[i])
                        tmp = (resi - functional.tau * 
                               (P(functional.grad(resi)) - 
                                functional.grad.adjoint(xvf[i-1]) * resi))       
                        tmp -= resii
                        tmp *= functional.alpha
                        out.assign(tmp)

                return IntermediateImageDerivative()

            else:
                raise ValueError('time index out of range')
    
    def spatial_vfield_gradient(self, i):
        if self.alpha == 0:
            return ZeroOperator(self.space, range=self.vf_space)
        else:
            functional = self
            i -= self.N
            if i == self.N-1:
                return ZeroOperator(self.space, self.vf_space)
            else:
                class VfieldDerivative(Operator):
                    def __init__(self):
                        super(VfieldDerivative, self).__init__(
                            functional.space, functional.vf_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        res = functional.residual(xim[i+1], xim[i], xvf[i])
                        tmp = functional.tau * functional.grad(xim[i+1]) * res
                        tmp *= functional.alpha
                        out.assign(tmp)

                return VfieldDerivative()

    def partial_gradient(self, i):  
        if i < self.N:
            return self.spatial_image_gradient(i)
        else:
            return self.spatial_vfield_gradient(i)
        

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2*self.N)])

class HuberL1OpticalFlowConstraint(Functional):

    def __init__(self, space, tau=1., alpha=1., grad=None, gamma=1e-7):
        self.N = round(len(space)/2)         # number of time steps

        if grad is None:
            grad = odl.Gradient(space[0], method='forward',
                                pad_mode='symmetric')
            grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
        else:
            grad = grad

        self.grad = grad
        self.space = space
        self.image_space = self.space[0]
        self.vf_space = self.space[self.N]
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.huber = Huber(self.image_space, self.gamma)
        super(HuberL1OpticalFlowConstraint,
              self).__init__(space=space, linear=False, grad_lipschitz=np.nan)

    def __call__(self, x):  # x is space time element im x vf
        ret = 0.
        xim = x[:self.N]
        xvf = x[self.N:]
        for i in range(self.N-1):
            ximi = xim[i]
            ximii = xim[i+1]
            xvfi = xvf[i]
            res = self.residual(ximii, ximi, xvfi)
            ret += self.huber(res)

        return self.alpha * ret

    
    def residual(self, im1, im2, vf):
        P = PointwiseInner(self.vf_space, vf)
        res = im1 - im2 + self.tau * P(self.grad(im1))

        return res

    def spatial_image_gradient(self, i):
        if self.alpha == 0:
            return ZeroOperator(self.space, range=self.image_space)
        else:
            functional = self
            if i == 0:
                class FirstImageDerivative(Operator):
                    def __init__(self):
                        super(FirstImageDerivative, self).__init__(
                                functional.space, functional.image_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        res = -functional.huber.gradient(
                                functional.residual(xim[1], xim[0], xvf[0]))
                        res *= functional.alpha
                        out.assign(res)

                return FirstImageDerivative()

            if i == self.N-1:
                class LastImageDerivative(Operator):
                    def __init__(self):
                        super(LastImageDerivative, self).__init__(
                                functional.space, functional.image_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        res = functional.residual(xim[-1], xim[-2], xvf[-2])
                        dhuber = functional.huber.gradient(res)
                        P = PointwiseInner(functional.vf_space, xvf[-2])
                        tmp = dhuber - functional.tau * (
                                P(functional.grad(dhuber)) - 
                                functional.grad.adjoint(xvf[-2]) * dhuber)
                        tmp *= functional.alpha
                        out.assign(tmp)

                return LastImageDerivative()

            if 0 < i < self.N-1:
                class IntermediateImageDerivative(Operator):
                    def __init__(self):
                        super(IntermediateImageDerivative, self).__init__(
                                functional.space, functional.image_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        resi = functional.residual(xim[i], xim[i-1], xvf[i-1])
                        resii = functional.residual(xim[i+1], xim[i], xvf[i])
                        dhuberi = functional.huber.gradient(resi)
                        dhuberii = functional.huber.gradient(resii)
                        P = PointwiseInner(functional.vf_space, xvf[i-1])
                        tmp = dhuberi - functional.tau * (
                                P(functional.grad(dhuberi)) - 
                                functional.grad.adjoint(xvf[i-1]) * dhuberi)
                        tmp -= dhuberii
                        tmp *= functional.alpha
                        out.assign(tmp)

                return IntermediateImageDerivative()

            else:
                raise ValueError('time index out of range')
    
    def spatial_vfield_gradient(self, i):
        if self.alpha == 0:
            return ZeroOperator(self.space, range=self.vf_space)
        else:
            functional = self
            i -= self.N
            if i == self.N-1:
                return ZeroOperator(self.space, self.vf_space)
            else:
                class VfieldDerivative(Operator):
                    def __init__(self):
                        super(VfieldDerivative, self).__init__(
                            functional.space, functional.vf_space)

                    def _call(self, x, out):
                        xim = x[:functional.N]
                        xvf = x[functional.N:]
                        res = functional.residual(xim[i+1], xim[i], xvf[i])
                        dhuber = functional.huber.gradient(res)
                        tmp = functional.tau * functional.grad(xim[i+1]) * dhuber
                        tmp *= functional.alpha
                        out.assign(tmp)

                return VfieldDerivative()

    def partial_gradient(self, i):  
        if i < self.N:
            return self.spatial_image_gradient(i)
        else:
            return self.spatial_vfield_gradient(i)
        

    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2*self.N)])

class AugmentedLagrangeTerm(Functional):
    def __init__(self, space, lagr_mult, tau, grad=None):    
        self.N = round(len(space)/2)
        self.space = space
        self.image_space = self.space[0]
        self.vf_space = self.space[self.N]
        self.lagr_mult = lagr_mult
        self.tau = tau
        if grad is None:
            grad = odl.Gradient(space[0], method='forward',
                                pad_mode='symmetric')
            grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
        else:
            grad = grad
        self.grad = grad    
        super(AugmentedLagrangeTerm,
              self).__init__(space=space, linear=False, grad_lipschitz=np.nan)
        
    def __call__(self, x):
        ret = 0
        xim = x[:self.N]
        xvf = x[self.N:]
        lam = self.lagr_mult
        for i in range(0,self.N-1):
            P = PointwiseInner(self.vf_space, xvf[i])
            ret += lam[i].inner(xim[i+1] - xim[i] + 
                      self.tau * P(self.grad(xim[i+1])))
            
        ret /= self.tau
        return ret
    
    def spatial_image_gradient(self,i):
        functional = self
        
        if i == 0:
            class FirstImageDerivative(Operator):
                def __init__(self):
                    super(FirstImageDerivative, self).__init__(
                            functional.space, functional.image_space)
                    
                def _call(self, x, out):
                    ret = -functional.lagr_mult[0]/functional.tau
                    out.assign(ret)
            return FirstImageDerivative()

        if i == self.N-1:
            class LastImageDerivative(Operator):
                def __init__(self):
                    super(LastImageDerivative, self).__init__(
                            functional.space, functional.image_space)

                def _call(self, x, out):
                    xvf = x[functional.N:]
                    P = PointwiseInner(functional.vf_space, xvf[-2])
                    ret = functional.lagr_mult[-2] - functional.tau * (
                            P(functional.grad(functional.lagr_mult[-2])) - 
                            functional.lagr_mult[-2] * 
                            functional.grad.adjoint(xvf[-2]))
                    ret /= functional.tau
                    out.assign(ret)
            return LastImageDerivative()

        if 0 < i < self.N-1:
            class IntermediateImageDerivative(Operator):
                def __init__(self):
                    super(IntermediateImageDerivative, self).__init__(
                            functional.space, functional.image_space)

                def _call(self, x, out):
                    xvf = x[functional.N:]
                    P = PointwiseInner(functional.vf_space, xvf[i-1])
                    ret = functional.lagr_mult[i-1] - functional.tau * (
                            P(functional.grad(functional.lagr_mult[i-1])) - 
                            functional.lagr_mult[i-1] * 
                            functional.grad.adjoint(xvf[i-1])) - functional.lagr_mult[i]                            
                    ret /= functional.tau
                    out.assign(ret)

            return IntermediateImageDerivative()

        else:
            raise ValueError('time index out of range')
        
    def spatial_vfield_gradient(self, i):
        functional = self
        i -= self.N
        if i == self.N-1:
            return ZeroOperator(self.space, self.vf_space)
        else:
            class VfieldDerivative(Operator):
                def __init__(self):
                    super(VfieldDerivative, self).__init__(
                        functional.space, functional.vf_space)

                def _call(self, x, out):
                    xim = x[:functional.N]
                    ret = functional.grad(xim[i+1]) * functional.lagr_mult[i]
                    out.assign(ret)

            return VfieldDerivative()   
        
    def partial_gradient(self, i):  
        if i < self.N:
            return self.spatial_image_gradient(i)
        else:
            return self.spatial_vfield_gradient(i)
        
    @property
    def gradient(self):
        return BroadcastOperator(*[self.partial_gradient(i) 
                                   for i in range(2*self.N)])    

class smooth_term_OF(Functional):
    def __init__(self, space, data, forward, tau, alpha_df, alpha_of, 
                 grad=None, huber=False, gamma=1e-7, aug_lagr=False, 
                 lagr_mult=None):
        self.N = round(len(space)/2)         # number of time steps\
        self.space = space
        self.image_space = self.space[0]
        self.space_time = ProductSpace(self.image_space, self.N)
        self.data = data
        self.forward = forward
        self.tau = tau
        self.alpha_df = alpha_df
        self.alpha_of = alpha_of
        self.grad = grad
        self.data_fit = DataFitL2TimeDep(self.space, self.data, self.forward,
                                         self.alpha_df)
        self.aug_lagr = aug_lagr
        if lagr_mult is None:
            self.lagr_mult = self.space_time.zero()
        else:
            self.lagr_mult = lagr_mult
        
        if huber is False:
            self.of_constr = L2OpticalFlowConstraint(self.space, self.tau, 
                                                 self.alpha_of, self.grad)
        else:
            self.of_constr = HuberL1OpticalFlowConstraint(self.space, self.tau, 
                                                 self.alpha_of, self.grad,
                                                 gamma)
        if aug_lagr is True:
            self.aug_lagr_term = AugmentedLagrangeTerm(self.space, 
                                                       self.lagr_mult,
                                                       self.tau, self.grad) 
            
        super(smooth_term_OF,
              self).__init__(space=space, linear=False, grad_lipschitz=np.nan)

    def __call__(self, x):
        ret = self.data_fit(x) + self.of_constr(x) 
        if self.aug_lagr is True:
            ret += self.aug_lagr_term(x)
        return ret            

    @property
    def gradient(self):
        grad_df = self.data_fit.gradient
        grad_of = self.of_constr.gradient
        if self.aug_lagr is True:
            grad_al = self.aug_lagr_term.gradient
            return BroadcastOperator(*[grad_df[i]+grad_of[i]+grad_al[i] 
                                   for i in range(2*self.N)])
        else:
            return BroadcastOperator(*[grad_df[i]+grad_of[i] 
                                   for i in range(2*self.N)])


class L1OpticalFlowConstraint(Functional):

    def __init__(self, space, alpha=1, grad=None):
        if not len(space) == 2:
            raise ValueError('Domain has not the right shape. Len=2 expected')

        if grad is None:
            grad = odl.Gradient(space[0], method='forward',
                                pad_mode='symmetric')
            grad.norm = 2 * np.sqrt(sum(1 / grad.domain.cell_sides**2))
        else:
            grad = grad

        self.alpha = alpha
        self.grad = grad
        self.image_space_time = space[0]
        self.image_space = self.image_space_time[0]
        self.vf_space_time = space[1]
        self.vf_space = self.vf_space_time[0]
        self.im_vf_space = ProductSpace(self.image_space, self.vf_space)
        self.N = len(self.image_space_time)         # number of time steps
        super(L1OpticalFlowConstraint,
              self).__init__(space=space, linear=False, grad_lipschitz=np.nan)

    def __call__(self, x):  # x is space time element im x vf
        ret = 0
        xim = x[0]
        xvf = x[1]
        for i in range(self.N-1):
            ximi = xim[i]
            ximii = xim[i+1]
            xvfi = xvf[i]
            res = self.residual(ximii, ximi, xvfi)
            ret += L1Norm(res.space)(res)

        return self.alpha * ret

    def residual(self, im1, im2, vf):
        P = PointwiseInner(self.vf_space, vf)
        res = im1 - im2 + P(self.grad(im1))

        return res

    @property
    def proximal(self):
        raise NotImplementedError('Not yet implemented')
