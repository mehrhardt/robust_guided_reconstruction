import numpy as np
from odl.solvers import L2NormSquared as odl_l2sq


class PALM():

    def __init__(self, f, g, ud_vars=None, x=None, niter=None, 
                 callback=None, L=None, tol=None):

        if x is None:
            x = f.domain.zero()
            
        if L is None:
            L = [1e2] * len(x)          
            
        if ud_vars is None:
            ud_vars = range(len(x))
            
        self.ud_vars = ud_vars    
        self.x = x
        self.f = f
        self.g = g
        self.etas = [1/1.1, 1.1] #[0.5, 10]
        self.L = L
        self.tol = tol
        self.callback = callback
        self.niter = 0
            
        self.dx = None
        self.x_old = None
        self.x_old2 = None

        self.g_old = None
        self.f_old = None

        if niter is not None:
            self.run(niter)

    def update_coordinate(self, i):        
        
        x = self.x
        x_old = self.x_old
        f = self.f
        g = self.g
        L = self.L
        
        BTsuccess = False
        if i==0:
            bt_vec = range(60)
        else:
            bt_vec = range(60)
        
        
        
        l2sq = odl_l2sq(x[i].space)
        df = f.gradient[i](x_old)
        
                    
        for bt in bt_vec:    #backtracking loop
            

            g[i].proximal(1/L[i])(x_old[i] - 1/L[i] * df, out=x[i])
            
            # backtracking on Lipschitz constants
            f_new = f(x)
            LHS1 = f_new
            
            self.dx[i] = x[i] - x_old[i]
            
            df_dxi = df.inner(self.dx[i])
            dxi_sq = l2sq(self.dx[i])
            
            RHS1 = self.f_old + df_dxi + L[i]/2 * dxi_sq   
            
            eps = 0e-4
            
            #print(i, bt, LHS1 - RHS1)
            if LHS1 > RHS1 + eps:
                L[i] *= self.etas[1]
                continue
            

            # proximal backtracking
            gi_new = g[i](x[i])
            LHS2 = gi_new
            RHS2 = self.g_old[i] - df_dxi - L[i]/2 * dxi_sq
            
            
            if LHS2 <= RHS2 + eps:                
                x_old[i][:] = x[i]
                self.f_old = f_new
                self.g_old[i] = gi_new
                L[i] *= self.etas[0]
                BTsuccess = True
                break
                                
            L[i] *= self.etas[1]    
    
                
        if BTsuccess is False:
            print('No step size found for variable {} after {} backtracking steps'.format(i, bt))
   

        if self.tol is not None:
            reldiff = dxi_sq/max(l2sq(x[i]), 1e-4)
                    
            if reldiff < self.tol:
                self.ud_vars.remove(i)
                print('Variable {} stopped updating'.format(i))

        
    def update(self):
        self.niter += 1
        
        if self.dx is None:
            self.dx = self.x.copy()
        if self.x_old is None:
            self.x_old = self.x.copy()
        if self.f_old is None:
            self.f_old = self.f(self.x_old)
        if self.g_old is None:
            self.g_old = [self.g[j](self.x_old[j]) for j in range(len(self.x))]
        
        
        for i in self.ud_vars:     #loop over variables
            
            self.update_coordinate(i)
            

    def run(self, niter=1):
        if self.tol is not None:
            if self.x_old2 is None:
                self.x_old2 = self.x.copy()
            l2sq = odl_l2sq(self.x.space)
            
        for k in range(niter):
            if self.x_old2 is None:
                self.x_old2 = self.x.copy()
            self.x_old2[:] = self.x
            self.update()
            
            dx = []
            for i in range(len(self.x)):
                 l2sq = odl_l2sq(self.x[i].space)   
                 dx.append(l2sq(self.dx[i])/max(l2sq(self.x[i]), 1e-4))
                 
            s = 'obj:{:3.2e}, f:{:3.2e}, g:{:3.2e}, diff:' + '{:3.2e} ' * len(self.x) + 'lip:' + '{:3.2e} ' * len(self.x)
            
            fx = self.f(self.x)
            gx = self.g(self.x)
            print(s.format(fx + gx, fx, gx, *dx, *self.L))

            if self.callback is not None:
                self.callback(self.x)

            if self.tol is not None:
                l2sq = odl_l2sq(self.x.space)   
                norm = l2sq(self.x_old2)
                if k > 1 and norm > 0:
                    crit = l2sq(self.x_old2-self.x)/norm
                else:
                    crit = np.inf
                    
                if crit < self.tol:
                    print('Stopped iterations with rel. diff. ',crit)
                    break
                else:
                    self.x_old2[:] = self.x

        return self.x


def fgp_dual(p, data, sigma, niter, grad, proj_C, proj_P, tol=None, **kwargs):
    callback = kwargs.pop('callback', None)
    if callback is not None and not callable(callback):
        raise TypeError('`callback` {} is not callable'.format(callback))

    factr = 1 / (grad.norm**2 * sigma)

    q = p.copy()
    x = data.space.zero()

    t = 1.

    if tol is None:
        def convergence_eval(p1, p2, k):
            return False
    else:
        def convergence_eval(p1, p2, k):
            return k > 5 and (p1 - p2).norm() / max(p1.norm(), 1) < tol

    pnew = p.copy()

    if callback is not None:
        callback(p)

    for k in range(niter):
        t0 = t
        grad.adjoint(q, out=x)
        proj_C(data - sigma * x, out=x)
        pnew = grad(x, out=pnew)
        pnew *= factr
        pnew += q

        proj_P(pnew, out=pnew)

        converged = convergence_eval(p, pnew, k)

        if not converged:
            # update step size
            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2.

            # calculate next iterate
            q[:] = pnew + (t0 - 1) / t * (pnew - p)

        p[:] = pnew

        if converged:
            t = None
            break

        if callback is not None:
            callback(p)

    # get current image estimate
    x = proj_C(data - sigma * grad.adjoint(p))

    return x
