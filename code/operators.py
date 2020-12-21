import numpy as np
import odl
from odl.operator import (Operator, IdentityOperator)
from scipy.ndimage import convolve as sp_convolve
from skimage.measure import block_reduce

###########################################
#################OPERATORS#################
###########################################



class RealFourierTransform(odl.Operator):
    
    def __init__(self, domain):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 1, 10) ** 2
        >>> F = myOperators.RealFourierTransform(X)
        >>> x = X.one()
        >>> y = F(x)
        """
        domain_complex = domain[0].complex_space
        self.fourier = odl.trafos.DiscreteFourierTransform(domain_complex)
        
        range = self.fourier.range.real_space ** 2
        
        super(RealFourierTransform, self).__init__(
                domain=domain, range=range, linear=True)
    
    def _call(self, x, out):
        Fx = self.fourier(x[0].asarray() + 1j * x[1].asarray())
        out[0][:] = np.real(Fx)
        out[1][:] = np.imag(Fx)
        
        out *= self.domain[0].cell_volume
                            
    @property
    def adjoint(self):
        op = self
        
        class RealFourierTransformAdjoint(odl.Operator):
    
            def __init__(self, op):        
                """TBC
                
                Parameters
                ----------
                TBC
                
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                """
                self.op = op
                
                super(RealFourierTransformAdjoint, self).__init__(
                        domain=op.range, range=op.domain, linear=True)
            
            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.adjoint(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)
                
                out *= self.op.fourier.domain.size
        
            @property
            def adjoint(self):
                return op
        
        return RealFourierTransformAdjoint(op)
    
    @property
    def inverse(self):
        op = self
        
        class RealFourierTransformInverse(odl.Operator):
    
            def __init__(self, op):        
                """TBC
                
                Parameters
                ----------
                TBC
                
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = A(x)
                >>> (A.inverse(y)-x).norm()
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = A(x)
                >>> (A.inverse(y)-x).norm()
                """
                self.op = op
                
                super(RealFourierTransformInverse, self).__init__(
                        domain=op.range, range=op.domain, linear=True)
            
            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.inverse(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)
                
                out /= self.op.fourier.domain.cell_volume
        
            @property
            def inverse(self):
                return op
        
        return RealFourierTransformInverse(op)

class UnitaryRealFourierTransform(odl.Operator):
    
    def __init__(self, domain):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 1, 10) ** 2
        >>> F = myOperators.RealFourierTransform(X)
        >>> x = X.one()
        >>> y = F(x)
        """
        domain_complex = domain[0].complex_space
        self.fourier = odl.trafos.DiscreteFourierTransform(domain_complex)
        
        range = self.fourier.range.real_space ** 2
        
        super(UnitaryRealFourierTransform, self).__init__(
                domain=domain, range=range, linear=True)
        
        self.factor = domain.one().norm() / 2
        self.factor = np.sqrt(self.domain[0].cell_volume / self.fourier.domain.size)
    
    def _call(self, x, out):
        Fx = self.fourier(x[0].asarray() + 1j * x[1].asarray())
        out[0][:] = np.real(Fx)
        out[1][:] = np.imag(Fx)
        
        out *= self.factor
                            
    @property
    def adjoint(self):
        op = self
        
        class RealFourierTransformAdjoint(odl.Operator):
    
            def __init__(self, op):        
                """TBC
                
                Parameters
                ----------
                TBC
                
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr(0, 2, 10) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
                >>> A = myOperators.RealFourierTransform(X)
                >>> x = odl.phantom.white_noise(A.domain)
                >>> y = odl.phantom.white_noise(A.range)
                >>> t1 = A(x).inner(y)
                >>> t2 = x.inner(A.adjoint(y))
                >>> t1 / t2
                """
                self.op = op
                
                super(RealFourierTransformAdjoint, self).__init__(
                        domain=op.range, range=op.domain, linear=True)
            
            def _call(self, x, out):
                y = x[0].asarray() + 1j * x[1].asarray()
                Fadjy = self.op.fourier.adjoint(y)
                out[0][:] = np.real(Fadjy)
                out[1][:] = np.imag(Fadjy)
                
                out /= self.op.factor
        
            @property
            def adjoint(self):
                return op
        
        return RealFourierTransformAdjoint(op)

    @property
    def inverse(self):
        """Inverse Fourier transform
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr(0, 2, 10) ** 2 
        >>> A = myOperators.RealFourierTransform(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> (A.inverse(A(x)) - x).norm()
        
        >>> import odl
        >>> import myOperators
        >>> X = odl.uniform_discr([-1, -1], [2, 1], [10, 30]) ** 2
        >>> A = myOperators.RealFourierTransform(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> (A.inverse(A(x)) - x).norm()
        """
        return self.adjoint

    
class Complex2Real(odl.Operator):
    
    def __init__(self, domain):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.cn(3)
        >>> J = myOperators.Complex2Real(X)
        >>> x = X.one()
        >>> y = J(x)
        
        >>> import odl
        >>> import myOperators
        >>> X = odl.cn(3)
        >>> A = myOperators.Complex2Real(X)
        >>> x = odl.phantom.white_noise(A.domain)
        >>> y = odl.phantom.white_noise(A.range)
        >>> t1 = A(x).inner(y)
        >>> t2 = x.inner(A.adjoint(y))
        """
        
        super(Complex2Real, self).__init__(domain=domain, 
                                           range=domain.real_space ** 2, 
                                           linear=True)
        
    def _call(self, x, out):
        out[0][:] = np.real(x)
        out[1][:] = np.imag(x)
                            
    @property
    def adjoint(self):
        return Real2Complex(self.range)
    
    
class Real2Complex(odl.Operator):
            
    def __init__(self, domain):
        """TBC

        Parameters
        ----------
        TBC

        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn(3) ** 2
        >>> J = myOperators.Real2Complex(X)
        >>> x = X.one()
        >>> y = J(x)
        """
        
        super(Real2Complex, self).__init__(domain=domain, 
             range=domain[0].complex_space, linear=True)
            
    def _call(self, x, out):
        out[:] = x[0].asarray() + 1j * x[1].asarray()
 
    @property
    def adjoint(self):
        return Complex2Real(self.range)    
    
    
def Magnitude(space):
    return odl.PointwiseNorm(space)


class Subsampling(odl.Operator):
    '''  '''
    def __init__(self, domain, range, margin=None):
        """TBC
        
        Parameters
        ----------
        TBC
        
        Examples
        --------
        >>> import odl
        >>> import myOperators
        >>> X = odl.rn((8, 8, 8))
        >>> Y = odl.rn((2, 2))  
        >>> S = myOperators.Subsampling(X, Y)
        >>> x = X.one()
        >>> y = S(x)
        """
        domain_shape = np.array(domain.shape)
        range_shape = np.array(range.shape)
        
        len_domain = len(domain_shape)
        len_range = len(range_shape)
        
        if margin is None:
            margin = 0
                
        if np.isscalar(margin):
            margin = [(margin, margin)] * len_domain

        self.margin = np.array(margin).astype('int')

        self.margin_index = []
        for m in self.margin:
            m0 = m[0]
            m1 = m[1]
            
            if m0 == 0:
                m0 = None
            
            if m1 == 0:
                m1 = None
            else:
                m1 = -m1

            self.margin_index.append((m0, m1))
                        
        if len_domain < len_range:
            ValueError('TBC')
        else:
            if len_domain > len_range:
                range_shape = np.append(range_shape, np.ones(len_domain - len_range))
                
            self.block_size = tuple(((domain_shape - np.sum(self.margin, 1)) / range_shape).astype('int'))

        super(Subsampling, self).__init__(domain=domain, range=range, 
                                          linear=True)
        
    def _call(self, x, out):
        m = self.margin_index
        if m is not None:
            if len(m) == 1:
                x = x[m[0][0]:m[0][1]]
            elif len(m) == 2:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1]]
            elif len(m) == 3:
                x = x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]]
            else:
                ValueError('TBC')
            
        out[:] = np.squeeze(block_reduce(x, block_size=self.block_size, func=np.mean))
                # block_reduce: returns Down-sampled image with same number of dimensions as input image.
                            
    @property
    def adjoint(self):
        op = self
            
        class SubsamplingAdjoint(odl.Operator):
            
            def __init__(self, op):
                """TBC
        
                Parameters
                ----------
                TBC
        
                Examples
                --------
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 15))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 15))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.rn((8, 8, 8))
                >>> Y = odl.rn((2, 2))  
                >>> S = myOperators.Subsampling(X, Y, margin=1)
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                
                >>> import odl
                >>> import myOperators
                >>> X = odl.uniform_discr([-1, -1, -.1], [1, 1, .1], (160, 160, 21))
                >>> Y = odl.uniform_discr([-1, -1], [1, 1], (40, 40))
                >>> S = myOperators.Subsampling(X, Y, margin=((0, 0),(0, 0),(3, 3)))
                >>> x = odl.phantom.white_noise(X)
                >>> y = odl.phantom.white_noise(Y)
                >>> S(x).inner(y) / x.inner(S.adjoint(y))
                """
                domain = op.range
                range = op.domain
                self.block_size = op.block_size
                self.margin = op.margin
                self.margin_index = op.margin_index
                
                x = range.zero()
                m = self.margin_index
                if m is not None:
                    if len(m) == 1:
                        x[m[0][0]:m[0][1]] = 1
                    elif len(m) == 2:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1]] = 1
                    elif len(m) == 3:
                        x[m[0][0]:m[0][1], m[1][0]:m[1][1], m[2][0]:m[2][1]] = 1
                    else:
                        ValueError('TBC')
                else:
                    x = range.one()
                
                self.factor = x.inner(range.one()) / domain.one().inner(domain.one())
                
                super(SubsamplingAdjoint, self).__init__(
                        domain=domain, range=range, linear=True)
                    
            def _call(self, x, out):
                for i in range(len(x.shape), len(self.block_size)):     
                    x = np.expand_dims(x, axis=i)
                                    
                if self.margin is None:
                    out[:] = np.kron(x, np.ones(self.block_size)) / self.factor                     
                else:      
                    y = np.kron(x, np.ones(self.block_size)) / self.factor                     
                    out[:] = np.pad(y, self.margin, mode='constant')

            @property
            def adjoint(self):
                return op
                    
        return SubsamplingAdjoint(self)
            

class Embedding_Affine(odl.Operator):
            def __init__(self,space_dom,space_range):
                self.space_affine = space_dom
                self.space_vf = space_range
                super(Embedding_Affine, self).__init__(domain=space_dom, 
                     range=space_range, linear=True)
            def _call(self, inp, out):
                shift = inp[0:2]
                matrix = inp[2:6]
                disp_vf = [
                        lambda x: matrix[0] * x[0] + matrix[1] * x[1] + shift[0],
                        lambda x: matrix[2] * x[0] + matrix[3] * x[1] + shift[1]]
                v = self.space_vf.element(disp_vf)
                out.assign(v)
                
            @property    
            def adjoint(self):
                op = self
                class AuxOp(odl.Operator):
                    def __init__(self):
                        super(AuxOp, self).__init__(domain=op.range, 
                             range=op.domain, linear=True)
                    def _call(self, phi, out):
                        phi0 = phi[0]
                        phi1 = phi[1]
                        space = phi0.space
                        aux_func0 = lambda x: x[0]
                        aux_func1 = lambda x: x[1]
                        
                        x0 = space.element(aux_func0)
                        x1 = space.element(aux_func1)
                        
                        mom00 = space.inner(x0,phi0)
                        mom10 = space.inner(x1,phi0)
                        mom01 = space.inner(x0,phi1)
                        mom11 = space.inner(x1,phi1)                        
                        
                        mean0 = space.inner(phi0,space.one())
                        mean1 = space.inner(phi1,space.one())                        
                        
                        ret = self.range.element([mean0, mean1, 
                                                  mom00, mom10, mom01, mom11])
                        out.assign(ret)
            
                return AuxOp()   
            
class Embedding_Affine_Rest(odl.Operator):
            def __init__(self,space_dom,space_range):
                self.space_affine = space_dom[0]
                self.space_rest = space_dom[1]
                self.space_vf = space_range
                super(Embedding_Affine_Rest, self).__init__(domain=space_dom, 
                     range=space_range, linear=True)
            def _call(self, inp, out):
                aff = inp[0]
                rest = inp[1]
                shift = aff[0:2]
                matrix = aff[2:6]
                disp_vf = [
                        lambda x: matrix[0] * x[0] + matrix[1] * x[1] + shift[0],
                        lambda x: matrix[2] * x[0] + matrix[3] * x[1] + shift[1]]
                v = self.space_vf.element(disp_vf) + rest
                out.assign(v)
                
            @property    
            def adjoint(self):
                op = self
                class AuxOp(odl.Operator):
                    def __init__(self):
                        super(AuxOp, self).__init__(domain=op.range, 
                             range=op.domain, linear=True)
                    def _call(self, phi, out):
                        phi0 = phi[0]
                        phi1 = phi[1]
                        space = phi0.space
                        aux_func0 = lambda x: x[0]
                        aux_func1 = lambda x: x[1]
                        
                        x0 = space.element(aux_func0)
                        x1 = space.element(aux_func1)
                        
                        mom00 = space.inner(x0,phi0)
                        mom10 = space.inner(x1,phi0)
                        mom01 = space.inner(x0,phi1)
                        mom11 = space.inner(x1,phi1)                        
                        
                        mean0 = space.inner(phi0,space.one())
                        mean1 = space.inner(phi1,space.one())                        
                        
                        retaff = op.space_affine.element([mean0, mean1, 
                                                  mom00, mom10, mom01, mom11])
                        retrest = phi
                        ret = self.range.element([retaff, retrest])
                        out.assign(ret)
            
                return AuxOp()        
            
##################################################
#################HELPER FUNCTIONS#################
##################################################

def get_cartesian_sampling(X, ufactor):
    sampling_array = np.zeros(X.shape)
    sampling_array[::ufactor, :] = 1
    sampling_array_vis = sampling_array.copy()
    sampling_array  = np.fft.fftshift(sampling_array) 
    sampling_points = np.nonzero(sampling_array)
    
    return odl.SamplingOperator(X, sampling_points), sampling_array_vis


def get_random_sampling(X, prob):
    sampling_array = np.random.choice([0,1], size=X.shape,p=[1-prob,prob])
    sampling_array_vis = sampling_array.copy()
    sampling_array  = np.fft.fftshift(sampling_array)               
    sampling_points = np.nonzero(sampling_array)
    
    return odl.SamplingOperator(X, sampling_points), sampling_array_vis

def get_radial_sampling(X, num_angles=10, block=0):
    sampling_array = np.zeros(X.shape)
    angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    center = [int(np.round(i/2)) for i in X.shape]
    
    for r in range(int(np.sqrt(2)*int(np.max(X.shape)/2))):
        for i in range(num_angles):
            phi = angles[i]
            idx = np.ceil([r*np.cos(phi), r*np.sin(phi)])
            idx = [int(i) for i in idx]
            
            glob_idx0 = center[0] + idx[0]
            glob_idx1 = center[1] + idx[1]
            
            if 0 <= glob_idx0 < X.shape[0] and 0 <= glob_idx1 < X.shape[1]:
                sampling_array[glob_idx0, glob_idx1] = 1
                
    
    sampling_array[center[0] - int(block/2) + 1 : center[0] + int(block/2) + 1, 
                   center[1] - int(block/2) + 1 : center[1] + int(block/2) + 1] = 1            
      
    sampling_array_vis = sampling_array.copy()                   
    sampling_array  = np.fft.fftshift(sampling_array)               
    sampling_points = np.nonzero(sampling_array)            
    
    return odl.SamplingOperator(X, sampling_points), sampling_array_vis
        


class Convolution(odl.Operator):

    def __init__(self, space, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(space, space, linear=True)

    def _call(self, x, out):
        sp_convolve(x, self.kernel, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            if self.domain.ndim == 2:
                kernel = np.fliplr(np.flipud(self.kernel.copy().conj()))
                kernel = self.kernel.space.element(kernel)
            else:
                raise NotImplementedError('"adjoint_kernel" only defined for '
                                          '2d kernels')

            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = Convolution(self.domain, kernel, origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbedding(odl.Operator):

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        sp_convolve(self.kernel, x, output=out.asarray(),
                    mode=self.boundary_condition, origin=self.origin)

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            origin = [0, ] * self.domain.ndim
            for i in range(self.domain.ndim):
                if np.mod(self.kernel.shape[i], 2) == 0:
                    origin[i] = -1

            for i in range(self.domain.ndim):
                if self.kernel.shape[i] < 3 and origin[i] == -1:
                    NotImplementedError('Shifted origins are only implemented '
                                        'for kernels of size 3 or larger.')

            self.__adjoint = ConvolutionEmbeddingAdjoint(self.range,
                                                         self.domain,
                                                         self.kernel,
                                                         origin, self)

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)


class ConvolutionEmbeddingAdjoint(odl.Operator):

    def __init__(self, domain, range, kernel, origin=None, adjoint=None):

        self.__kernel = kernel
        self.__origin = origin
        self.__adjoint = adjoint
        self.__boundary_condition = 'constant'
        self.__scale = kernel.space.domain.volume / len(kernel)

        super().__init__(domain, range, linear=True)

    def _call(self, x, out):
        if not self.domain.ndim == 2:
            raise NotImplementedError('adjoint only defined for 2d domains')

        out_a = out.asarray()
        x_a = x.asarray()
        k_a = self.kernel.asarray()

        n = x.shape
        s = out.shape[0] // 2, out.shape[1] // 2

        for i in range(out_a.shape[0]):

            if n[0] > 1:
                ix1, ix2 = max(i - s[0], 0), min(n[0] + i - s[0], n[0])
                ik1, ik2 = max(s[0] - i, 0), min(n[0] - i + s[0], n[0])
            else:
                ix1, ix2 = 0, 1
                ik1, ik2 = 0, 1

            for j in range(out_a.shape[1]):
                if n[1] > 1:
                    jx1, jx2 = max(j - s[1], 0), min(n[1] + j - s[1], n[1])
                    jk1, jk2 = max(s[1] - j, 0), min(n[1] - j + s[1], n[1])
                else:
                    jx1, jx2 = 0, 1
                    jk1, jk2 = 0, 1

                out_a[i, j] = np.sum(x_a[ix1:ix2, jx1:jx2] *
                                     k_a[ik1:ik2, jk1:jk2])

        out *= self.__scale

    @property
    def kernel(self):
        return self.__kernel

    @property
    def origin(self):
        if self.__origin is None:
            self.__origin = [0, ] * self.domain.ndim

        return self.__origin

    @property
    def boundary_condition(self):
        return self.__boundary_condition

    @property
    def adjoint(self):
        if self.__adjoint is None:
            NotImplementedError('Can only be called as an "adjoint" of '
                                '"ConvolutionEmbedding".')

        return self.__adjoint

    def __repr__(self):
        """Return ``repr(self)``."""
        return '{}({!r}, {!r}, {!r}, {!r})'.format(
            self.__class__.__name__, self.domain, self.kernel, self.origin,
            self.boundary_condition)
