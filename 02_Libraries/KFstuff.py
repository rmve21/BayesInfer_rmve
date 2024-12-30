from numpy import transpose, zeros, ndarray, reshape, isscalar
from numpy import diag, dot, log, eye, delete, isnan, mean, std
from numpy import array
from numpy.linalg import inv, det, cholesky, solve, norm
from numpy.random import normal, uniform, randn
from numpy import sum as suma
from numpy import sqrt as sqrta
from math import ceil, sqrt
from matplotlib import pyplot as plt

def KFplus(Y, mod, ops=None):

    # Check dimensions
    YY = Y.T
    np, n = YY.shape
    nZ = nH = nT = nQ = nR = nc = 1
    if len(mod['Z'].shape)==3:
        aZ, bZ, nZ = mod['Z'].shape
    else:
        aZ, bZ = mod['Z'].shape
    if len(mod['H'].shape)==3:
        aH, bH, nH = mod['H'].shape
    else:
        aH, bH = mod['H'].shape
    if len(mod['T'].shape)==3:
        aT, bT, nT = mod['T'].shape
    else:
        aT, bT = mod['T'].shape
    if len(mod['Q'].shape)==3:
        aQ, bQ, nQ = mod['Q'].shape
    else:
        aQ, bQ = mod['Q'].shape  

    # Default setup for matrix R in the state equation
    if 'R' not in mod:
        mod['R'] = eye(aT)
    if len(mod['R'].shape)==3:
        aR, bR, nR = mod['R'].shape
    else:
        aR, bR = mod['R'].shape  

    ns = bZ # Dimension of state vector
    # Dimension of perturbation vector in state equation
    nr = bR if not isscalar(mod['R']) else bQ

    # Check comfortability in modtem matrices
    if aZ != np:
        raise ValueError('Wrong dimension: Z should be (p x s)')
    if aH != np or bH != np:
        raise ValueError('Wrong dimension: H should be (p x p)')
    if aT != ns or bT != ns:
        raise ValueError('Wrong dimension: T should be (s x s)')
    if not isscalar(mod['R']):
        if aQ != nr or bQ != nr:
            raise ValueError('Wrong dimension: Q should be (r x r)')
    if not isscalar(mod['R']):
        if aR != ns:
            raise ValueError('Wrong dimension: R should be (s x r)')

    # Deterministic vectors in the measurement equation
    if 'd' in mod:
        flagd = True
        flagX = False
        ad, bd, nd = mod['d'].shape
        if ad != np or bd != 1:
            raise ValueError('Wrong dimension: d should be (p x 1)')
    else:
        flagd = False
        flagX = False
        # Regressors vectors in the measurement equation
        if 'X' in mod:
            flagX = True
            if 'D' not in mod:
                raise ValueError('Matrix D not specified')
            XX = mod['X'].T
            kx, nx = XX.shape
            rowsD, colsD = mod['D'].shape
            if kx != colsD:
                raise ValueError('Wrong dimension: X should be (n x kx) and D should be (p x kx)')
            if np != rowsD:
                raise ValueError('Wrong dimension: D should be (p x kx)')

    # Deterministic vectors in the state equation
    if 'c' in mod:
        flagc = True
        flagW = False
        if len(mod['c'].shape)==3:
            ac, bc, nc = mod['c'].shape
        else:
            ac, bc = mod['c'].shape
        if ac != ns or bc != 1:
            raise ValueError('Wrong dimension: c should be (s x 1)')
    else:
        flagc = False
        flagW = False
        # Regressors vectors in the state equation
        if 'W' in mod:
            flagW = True
            if 'C' not in mod:
                raise ValueError('Matrix C not specified')
            WW = mod['W'].T
            kw, nw = WW.shape
            rowsC, colsC = mod['C'].shape
            if kw != colsC:
                raise ValueError('Wrong dimension: W should be (s x kw) and C should be (s x kw)')
            if ns != rowsC:
                raise ValueError('Wrong dimension: C should be (s x kw)')

    # Defaults and options
    if ops is not None:
        compute_updated = ops.get('compute_updated',True)
        compute_smoothed = ops.get('compute_smoothed',True)
        compute_stderrors = ops.get('compute_stderrors',True)
        only_likelihood = ops.get('only_likelihood',False)
        compute_grad = ops.get('compute_grad',False)
        nf = ops.get('nf',0)
        a_initial = ops.get('a_initial',zeros(ns))
        P_initial = ops.get('P_initial',eye(ns) * 1e+10)
        removefirst = ops.get('removefirst',5)
        computeP = ops.get('computeP',True)
        
    else:
        compute_updated, compute_smoothed, compute_stderrors = True, True, True
        only_likelihood, compute_grad, computeP, nf = False, False, True, 0
        a_initial,  P_initial = zeros(ns), eye(ns)*1e+10
        removefirst = 5

    if only_likelihood:
        compute_updated = compute_smoothed = compute_stderrors = False

    # Check if forecasting is possible
    if flagd and nd > 1:
        if nd < (n + nf):
            raise ValueError('Wrong dimension: d does not contain data for forecasting')

    if flagX and nx < (n + nf):
        raise ValueError('Wrong dimension: X does not contain data for forecasting')

    if flagc and nc > 1:
        if nc < (n + nf):
            raise ValueError('Wrong dimension: c does not contain data for forecasting')

    if flagW and nw < (n + nf):
        raise ValueError('Wrong dimension: W does not contain data for forecasting')

    # Check for missing values
    miss = isnan(Y)
    all_miss = suma(miss, axis=1) == np
    any_miss = suma(miss, axis=1) > 0
    some_miss = any_miss & ~all_miss

    # Check if system is time-invariant
    if all([nZ == 1, nH == 1, nT == 1, nQ == 1, nR == 1]):
        time_invariant = True
    else:
        time_invariant = False

    # *********************************
    # Generate objects to store computations
    # KF: Prediction
    a = zeros((ns, n + 1))
    P = zeros((ns, ns, n + 1))
    L = zeros((ns, ns, n + 1))
    v = zeros((np, n))
    F = zeros((np, np, n))
    K = zeros((ns, np, n))
    invF = zeros((np, np, n))
    if compute_stderrors:
        sd_v = zeros((np, n))

    # KF: Updating
    if compute_updated:
        au = zeros((ns, n))
        Pu = zeros((ns, ns, n))
        if compute_stderrors:
            sd_au = zeros((ns, n))

    # KF: Smoothing
    if compute_smoothed:
        r = zeros((ns, n))
        N = zeros((ns, ns, n))
        as_ = zeros((ns, n))
        Ps = zeros((ns, ns, n))
        if compute_stderrors:
            sd_as = zeros((ns, n))
        ee = zeros((np, n))
        Ve = zeros((np, np, n))
        uu = zeros((np, n))
        DD = zeros((np, np, n))
        hh = zeros((nr, n))
        Vh = zeros((nr, nr, n))

    # KF: Forecasting
    if nf > 0:
        af = zeros((ns, nf))
        Pf = zeros((ns, ns, nf))
        yf = zeros((np, nf))
        Ff = zeros((np, np, nf))
        if compute_stderrors:
            sd_yf = zeros((np, nf))
            sd_af = zeros((ns, nf))

    a[:, 0] = a_initial
    P[:, :, 0] = P_initial

    result = mod

    Dx = 0
    Cw = 0

    if time_invariant:
        ZZ = mod['Z']
        HH = mod['H']
        TT = mod['T']
        RR = mod['R']
        QQ = mod['Q']

    minuslogL = 0
    logLi = zeros((n, 1))

    for t in range(1, n + nf + 1):
        if not time_invariant:
            ZZ = mod['Z'] if nZ == 1 else mod['Z'][:, :, t - 1]
            HH = mod['H'] if nH == 1 else mod['H'][:, :, t - 1]
            TT = mod['T'] if nT == 1 else mod['T'][:, :, t - 1]
            RR = mod['R'] if nR == 1 else mod['R'][:, :, t - 1]
            QQ = mod['Q'] if nQ == 1 else mod['Q'][:, :, t - 1]

        if flagd:
            Dx = mod['d'][:, :, t - 1] if nd > 1 else mod['d']
        elif flagX:
            Dx = mod['D'].dot(XX[:, t - 1])

        if flagc:
            Cw = mod['c'][:, :, t - 1] if nc > 1 else mod['c']
        elif flagW:
            Cw = mod['C'].dot(WW[:, t - 1])

        if t <= n:
            # In-sample operations: Prediction and filtering
            yy = YY[:, t - 1] - Dx

            if all_miss[t - 1]:
                v[:, t - 1] = 0
            elif some_miss[t - 1]:
                W = eye(np)
                W = delete(W,miss[t - 1, :], axis=0)
                yy[miss[t - 1, :]] = 0
                v[:, t - 1] = dot(dot(W.T,W),yy - dot(ZZ,a[:, t - 1]))
                computeP = True
            else:
                v[:, t - 1] = yy - dot(ZZ,a[:, t - 1])
                W = eye(np)

            if computeP:
                MM = dot(P[:, :, t - 1],ZZ.T)
                if not all_miss[t - 1]:
                    F[:, :, t - 1] = dot(dot(dot(W.T,W),dot(ZZ,MM) + HH),dot(W.T,W))
                    invF[:, :, t - 1] = dot(W.T,( solve(dot(dot(W,F[:, :, t-1]),W.T), W) ))
                    K[:, :, t - 1] = dot(dot(TT,MM),invF[:, :, t - 1])
                # Recursion for the conditional MSE
                L[:, :, t - 1] = TT - dot(K[:, :, t - 1],ZZ)
                P[:, :, t] = dot(dot(TT,P[:, :, t - 1]),L[:, :, t - 1].T) + dot(dot(RR,QQ),RR.T)
            else:
                # Filter converged to a steady state rule
                if not any_miss[t - 1]:
                    F[:, :, t - 1] = F[:, :, t - 2]
                    invF[:, :, t - 1] = invF[:, :, t - 2]
                    L[:, :, t - 1] = L[:, :, t - 2]
                    K[:, :, t - 1] = K[:, :, t - 2]
                    P[:, :, t] = P[:, :, t - 1]
                else:
                    computeP = True
                    P[:, :, t] = dot(dot(TT,P[:, :, t - 1]),L[:, :, t - 1].T) + dot(dot(RR,QQ),RR.T)
            # Recursion for the conditional mean
            a[:, t] = reshape(Cw + dot(TT,reshape(a[:, t - 1],(ns,1))) + dot(K[:, :, t - 1],reshape(v[:, t - 1],(np,1))),(ns,))

            # Log-likelihood evaluation
            dF = det(dot(dot(W,F[:, :, t - 1]),W.T))
            if dF > 0:
                logLi[t - 1, :] = -0.5 * (log(dF) + dot(dot(v[:, t - 1].T,invF[:, :, t - 1]),v[:, t - 1]))

            if t > (removefirst + 1):
                minuslogL = minuslogL - logLi[t - 1, :] / n

            # Check if the filter has reached the steady state
            if time_invariant and (t > 1) and computeP and not any_miss[t - 1]:
                if norm(P[:, :, t] - P[:, :, t - 1]) < 1e-8:
                    computeP = False

            # Standard errors
            if compute_stderrors:
                sd_v[:, t - 1] = sqrta(diag(F[:, :, t - 1]))

            # Updated quantities
            if compute_updated:
                au[:, t - 1] = a[:, t - 1] + dot(dot(MM,invF[:, :, t - 1]),v[:, t - 1])
                Pu[:, :, t - 1] = P[:, :, t - 1] - dot(dot(MM,invF[:, :, t - 1]),MM.T)
                if compute_stderrors:
                    sd_au[:, t - 1] = sqrta(diag(Pu[:, :, t - 1]))

        else:
            # Forecasting
            h = t - n
            if h == 1:
                if compute_updated:
                    an = au[:, n - 1]
                    Pn = Pu[:, :, n - 1]
                else:
                    an = a[:, n - 1] + dot(dot(MM,invF[:, :, n - 1]),v[:, n - 1])
                    Pn = P[:, :, n - 1] - dot(dot(MM,invF[:, :, n - 1]),MM.T)
                
                af[:, h-1] = Cw + dot(TT,an)
                Pf[:, :, h-1] = dot(dot(TT,Pn),TT.T) + dot(dot(RR,QQ),RR.T)

            else:
                af[:,    h-1] = Cw + TT*af[:, h - 2]
                Pf[:, :, h-1] = dot(dot(TT,Pf[:, :, h - 1]),TT.T) + dot(dot(RR,QQ),RR.T)
            
            yf[:, h-1] = dot(ZZ,af[:, h-1]) + Dx
            Ff[:, :, h-1] = dot(dot(ZZ,Pf[:,:,h-1]),ZZ.T) + HH
            if compute_stderrors:
                sd_af[:,h-1] = sqrta(diag(Pf[:, :, h-1]))
                sd_yf[:,h-1] = sqrta(diag(Ff[:, :, h-1]))


     # Storing the results
    if not only_likelihood:
        result = {'a_p': a.T, 'Sa_p': P, 'e': v.T, 'Se': F, 'invSe': invF, 'K': K}

    if compute_stderrors:
        result['e_sd'] = sd_v.T

    result['logLi'] = logLi

    if compute_updated:
        result['a_u'] = au.T
        result['Sa_u'] = Pu
        if compute_stderrors:
            result['a_u_std'] = sd_au.T

    if nf>0:
        result['a_f'] = af.T
        result['Sa_f'] = Pf
        result['y_f'] = yf.T
        result['Sy_f'] = Ff
        if compute_stderrors:
            result['y_f_std'] = sd_yf.T
            result['a_f_std'] = sd_af.T

    # Smoother
    if compute_smoothed:
        if compute_grad:
            Sr = zeros((ns, ns))

        for t in range(n, 0, -1):
            if nZ > 1:
                ZZ = mod.Z[:, :, t - 1]
            if t > 1:
                r[:, t - 2] = dot(dot(ZZ.T,invF[:, :, t - 1]),v[:, t - 1]) + dot(L[:, :, t - 1].T,r[:, t - 1])
                N[:, :, t - 2] = dot(dot(ZZ.T,invF[:, :, t - 1]),ZZ) + dot(dot(L[:, :, t - 1].T,N[:, :, t - 1]),L[:, :, t - 1])

                as_[:, t - 1] = a[:, t - 1] + dot(P[:, :, t - 1],r[:, t - 2])
                Ps[:, :, t - 1] = P[:, :, t - 1] - dot(dot(P[:, :, t - 1],N[:, :, t - 2]),P[:, :, t - 1].T)

                if compute_grad:
                    Sr += 0.5 * (dot(r[:, t - 2],r[:, t - 2].T) - N[:, :, t - 2])

            else:
                r0 = dot(dot(ZZ.T,invF[:, :, t - 1]),v[:, t - 1]) + dot(L[:, :, t - 1].T,r[:, t - 1])
                N0 = dot(dot(ZZ.T,invF[:, :, t - 1]),ZZ) + dot(dot(L[:, :, t - 1].T,N[:, :, t - 1]),L[:, :, t - 1])

                as_[:, t - 1] = a[:, t - 1] + dot(P[:, :, t - 1],r0)
                Ps[:, :, t - 1] = P[:, :, t - 1] - dot(dot(P[:, :, t - 1],N0),P[:, :, t - 1].T)

            if compute_stderrors:
                sd_as[:, t - 1] = sqrta(diag(Ps[:, :, t - 1]))

        # Storing the results
        result['a_s'] = as_.T
        result['Sa_s'] = Ps
        if compute_stderrors:
            result['a_s_std'] = sd_as.T

    # Disturbance Smoother
    if compute_smoothed:
        if compute_grad:
            Su = zeros((np, np))

        for t in range(n):
            if nH > 1:
                HH = mod.H[:, :, t]
            if nQ > 1:
                QQ = mod.Q[:, :, t]
            if nR > 1:
                RR = mod.R[:, :, t]

            uu[:, t] = dot(invF[:, :, t],v[:, t]) - dot(K[:, :, t].T,r[:, t])
            DD[:, :, t] = invF[:, :, t] + dot(dot(K[:, :, t].T,N[:, :, t]),K[:, :, t])

            ee[:, t] = dot(HH,uu[:, t])
            Ve[:, :, t] = HH - dot(dot(HH,DD[:, :, t]),HH)

            hh[:, t] = dot(dot(QQ,RR.T),r[:, t])
            Vh[:, :, t] = QQ - dot(dot(dot(dot(QQ,RR.T),N[:, :, t]),RR),QQ)

            if compute_grad:
                Su += 0.5 * (dot(uu[:, t],uu[:, t].T) - DD[:, :, t])

        # Storing the results
        # result['u'] = uu.T; result['D'] = DD
        result['eps'] = ee.T; result['Veps'] = Ve
        result['eta'] = hh.T; result['Veta'] = Vh

        if compute_grad:
            result['Su'] = Su
            result['Sr'] = Sr

        result['minuslogL'] = minuslogL.item()
        
    if only_likelihood:
        return minuslogL.item()
    else:
        return result
    
def SS_simul(mod, nobs):

    # Set minimo de matrices
    Z = mod['Z']
    T = mod['T']
    Q = mod['Q']

    # Verificando dimensiones
    ny, _ = Z.shape
    ns, _ = T.shape
    mQ, nQ = Q.shape

    # Regresores exogenos
    if 'xdata' in mod:
        flagX = True
        flagB = False
        if 'B' not in mod:
            raise ValueError('"X" was specified but "B"!')

        x = mod('xdata').T
        B = mod('B')
        mx = x.shape[0]
        [mB, nB] = B.shape
        if not mx == nB:
            raise ValueError('Wrong dimension: "X" should be (n_obs x nx) and "B" should be (ny x nx)')

        if not ny == mB:
            raise ValueError('Wrong dimension: B should be (ny x nx)')

    else:
        flagX = False
        if 'B' in mod:
            flagB = True
            B = mod['B']
            [mB, nB] = B.shape
            if not (nB == 1 and ny == mB):
                raise ValueError('Wrong dimension: "B" should be a (ny x 1) vector')
 
        else:
            flagB = False

    # H: varcov de errores de medida
    H = mod.get('H', zeros((ny, ny)))

    # c: intercepto en sistema de estados
    c = mod.get('c', zeros((ns, 1)))

    # R: proporción shocks en sistema de estados
    R = mod.get('R', eye(ns))

    mH, nH = H.shape
    mR, nR = R.shape

    if R.size > 1:
        nr = nR
    else:
        nr = nQ

    if not (mH == ny and nH == ny):
        raise ValueError('Wrong dimension: H should be (ny x ny)')

    if R.size == 1 and not (mQ == nr and nQ == nr):
        raise ValueError('Wrong dimension: Q should be (nr x nr)')

    if R.size > 1 and not mR == ns:
        raise ValueError('Wrong dimension: R should be (ns x nr)')

    # reservando espacio
    a = zeros((ns, nobs))
    y = zeros((ny, nobs))

    # Punto inicial
    a0 = reshape(mod['alpha0'], (ns, 1))
    sqrtQ = cholesky(Q)
    sqrtH = cholesky(H)

    for tt in range(nobs):
        if flagX:
            Bx = dot(B, reshape(x[:, tt], (mx, 1)))
        elif flagB:
            Bx = B
        else:
            Bx = 0

        mu = dot(sqrtQ, normal(0, 1, (nr, 1)))
        Rmu = dot(R, mu)
        ep = dot(sqrtH, normal(0, 1, (ny, 1)))
        if tt == 0:
            aux0 = dot(T, a0)
            a[:, tt] = reshape(c + aux0 + Rmu, (ns,))
        else:
            aux0 = dot(T, reshape(a[:, tt-1], (ns, 1)))
            a[:, tt] = reshape(c + aux0 + Rmu, (ns,))

        aux0 = dot(Z, reshape(a[:, tt], (ns, 1)))
        y[:, tt] = reshape(aux0 + Bx + ep, (ny,))

    return a, y


def get_MHsim(n, b, MinusLogPost, x0, LTOmega, scale, data, u_prior, s2_prior):

    k = LTOmega.shape[0]
    xsim = zeros(shape=(k, n+b))
    aceptance = 0

    toolbar_width = 50
    step = max(1, int((n + b) / toolbar_width))
    print("%s]\n[" % (" " * toolbar_width), end="", flush=True)
    for ii in range(n+b):
        if ii == 0:
            xsim[:, ii] = x0.flatten()
            x_old = xsim[:, ii].reshape((k,))
            fx_old = -1*MinusLogPost(x_old, data, u_prior, s2_prior) 
        else:
            u = log(uniform(0, 1))
            x_new = x_old + (scale*dot(LTOmega, randn(k))).flatten() # Candidate
            # print(x_new)
            mfx_new, aux0 = MinusLogPost(x_new, data, u_prior, s2_prior, False)
            fx_new = -1*mfx_new
            alpha = min(fx_new-fx_old, 0)

            if ii == b:
                a_s = aux0['a_s']
                Tt, nv = a_s.shape
                smooth_sim = zeros(shape=(Tt, nv, n))

            a_s = aux0['a_s']
            flagEstable = True
            if 'solución' in aux0:
                flagEstable = aux0['solución']['Estable']
            # print(ii)
            # print('fx_new=',fx_new)
            # print('fx_old=',fx_old)
            # print('alpha=',alpha)
            # print('u=',u)
            # print('')
            if u <= alpha and not isnan(x_new).any() and not isnan(a_s).any() and flagEstable:
                fx_old = fx_new
                x_old = x_new
                xsim[:, ii] = x_new
                aceptance += 1
                if ii >= b:
                    smooth_sim[:, :, ii-b] = a_s
            else:
                xsim[:, ii] = xsim[:, ii-1]
                if ii >= b:
                    smooth_sim[:, :, ii-b] = smooth_sim[:, :, ii-b-1]

        if (ii % step) == 0:
            print(f"\r{'-'*(ii//step)} R. Acc. = {(aceptance/(ii+1))*100:.2f}", end="")
  

    print(f"\r{'-'*(ii//step)}]")
    xsim = delete(xsim, range(b), 1)
    return xsim, (aceptance/(n+b))*100, smooth_sim


def modecheck(fcn0, x, xup, xdown, data, others=dict()):

    # u_prior, s2_prior, fcn1
    flagPost = 'fcn1' in others
    if flagPost:
        fcn1 = others['fcn1']
        flagprior = 'u_prior' in others
        if flagprior:
            u_prior = others['u_prior']
            s2_prior = others['s2_prior']

    k = x.size
    km = ceil(sqrt(k))

    fig, axs = plt.subplots(km, km)
    vv = 0
    for r in range(km):
        for c in range(km):
            if vv < k:
                p, pu, pd = x[vv], xup[vv], xdown[vv]
                gridp = pd + (pu-pd)*array(list(range(25)))/25
                llk = zeros((25, 1))

                if flagPost:
                    post = zeros((25, 1))

                for ev in range(25):
                    x1 = x.copy()
                    x1[vv] = gridp[ev]
                    llk[ev] = fcn0(x1, data)
                    
                    if flagPost:
                        if flagprior:
                            post[ev] = fcn1(x1, data, u_prior, s2_prior, True)
                        else:
                            post[ev] = fcn1(x1, data)

                llk_m = mean(llk)
                llk_s = std(llk)
                llk = (llk-llk_m)/llk_s
                axs[r, c].plot(gridp, llk, 'g')
                axs[r, c].axvline(p, color='red', lw=1, ls='--')

                if flagPost:
                    post_m = mean(post)
                    post_s = std(post)
                    post = (post-post_m)/post_s
                    axs[r, c].plot(gridp, post, 'r')

            vv += 1

    plt.tight_layout()
    return


## Primera versión simple de KF
# Filtro de Kalman básico
def Basic_KF(data, mod, flagEstima=False):

    # Set minimo de matrices
    Z = mod['Z']
    T = mod['T']
    Q = mod['Q']
    y = transpose(data)

    # Verificando dimensiones
    ny, nobs = y.shape
    mZ, nZ = Z.shape
    mT, nT = T.shape
    mQ, nQ = Q.shape

    # Regresores exogenos
    if 'xdata' in mod:
        flagX = True
        flagB = False
        if 'B' not in mod:
            print('Matrix B not specified')
            return mod

        x = transpose(mod('xdata'))
        B = mod('B')
        [mx, nx] = x.shape
        [mB, nB] = B.shape
        if not mx == nB:
            print('Wrong dimension:')
            print('X should be (n_obs x nx) and B should be (ny x nx)')
            return mod

        if not ny == mB:
            print('Wrong dimension: B should be (ny x nx)')
            return mod

    else:
        flagX = False
        if 'B' in mod:
            flagB = True
            B = mod('B')
            [mB, nB] = B.shape
            if nB > 1 or (nB == 1 and not ny == nB):
                print('Wrong dimension:')
                print('B should be a (ny x 1) vector')
                return mod
        else:
            flagB = False

    # H: varcov de errores de medida
    if 'H' not in mod:
        H = zeros(shape=(ny, ny))
        mod['H'] = H
    else:
        H = mod['H']

    # c: intercepto en sistema de estados
    if 'c' not in mod:
        c = zeros(shape=(mT, 1))
        mod['c'] = c
    else:
        c = mod['c']

    # R: proporción shocks en sistema de estados
    if 'R' not in mod:
        R = eye(mT)
        mod['R'] = R
    else:
        R = mod['R']

    mH, nH = H.shape
    mR, nR = R.shape

    ns = nZ
    if R.size > 1:
        nr = nR
    else:
        nr = nQ

    if not mZ == ny:
        print('Wrong dimension: Z should be (ny x ns)')
        return mod

    if not (mH == ny and nH == ny):
        print('Wrong dimension: H should be (ny x ny)')
        return mod

    if not (mT == ns and nT == ns):
        print('Wrong dimension: T should be (ns x ns)')
        return mod

    if R.size == 1:
        if not (mQ == nr and nQ == nr):
            print('Wrong dimension: Q should be (nr x nr)')
            return mod

    if R.size > 1:
        if not mR == ns:
            print('Wrong dimension: R should be (ns x nr)')
            return mod

    # reservando espacio
    a_p = ndarray(shape=(ns, nobs))
    Sa_p = ndarray(shape=(ns, ns, nobs))
    y_p = ndarray(shape=(ny, nobs))
    e_p = ndarray(shape=(ny, nobs))
    Se_p = ndarray(shape=(ny, ny, nobs))

    K = ndarray(shape=(ns, ny, nobs))

    a_u = ndarray(shape=(ns, nobs))
    Sa_u = ndarray(shape=(ns, ns, nobs))

    minuslogL = 0
    logLi = zeros(shape=(nobs,))

    # Punto inicial
    a_u0 = reshape(mod['alpha0'], (ns, 1))
    Sa_u0 = mod['Sigma_a0']

    RQR = dot(dot(R, Q), transpose(R))

    for tt in range(nobs):
        if flagX:
            Bx = dot(B, reshape(x[:, tt], (mx, 1)))
        elif flagB:
            Bx = B
        else:
            Bx = 0

        # Prediction
        if tt == 0:
            aux0 = dot(T, a_u0)
            a_p[:, tt] = reshape(c + aux0, (ns,))
            aux0 = dot(dot(T, Sa_u0), transpose(T))
            Sa_p[:, :, tt] = aux0 + RQR
        else:
            aux0 = dot(T, reshape(a_u[:, tt-1], (ns, 1)))
            a_p[:, tt] = reshape(c + aux0, (ns,))
            aux0 = dot(dot(T, Sa_u[:, :, tt-1]), transpose(T))
            Sa_p[:, :, tt] = aux0 + RQR

        aux0 = dot(Z, reshape(a_p[:, tt], (ns, 1)))
        y_p[:, tt] = reshape(aux0 + Bx, (ny,))
        e_p[:, tt] = reshape(y[:, tt]-y_p[:, tt], (ny,))
        Se_p[:, :, tt] = dot(dot(Z, Sa_p[:, :, tt]), transpose(Z)) + H

        # Ganancia de Kalman
        iSe_p = inv(Se_p[:, :, tt])
        K[:, :, tt] = dot(dot(Sa_p[:, :, tt], transpose(Z)), iSe_p)

        # Likelihood
        dSe = det(Se_p[:, :, tt])
        if dSe > 0:
            evec = reshape(e_p[:, tt], (ny, 1))
            logLi[tt] = - 0.5*(log(dSe)
                               + dot(dot(transpose(evec), iSe_p), evec))

        minuslogL -= logLi[tt]/nobs

        # Actualización
        aux0 = reshape(dot(K[:, :, tt], reshape(e_p[:, tt], (ny, 1))), (ns,))
        a_u[:, tt] = a_p[:, tt] + aux0
        aux0 = dot(dot(K[:, :, tt], Z), Sa_p[:, :, tt])
        Sa_u[:, :, tt] = Sa_p[:, :, tt] - aux0

    if flagEstima:
        return minuslogL.item()
    else:
        # Memoria para smoothed
        a_s = ndarray(shape=(ns, nobs))
        Sa_s = ndarray(shape=(ns, ns, nobs))
        a_s_std = ndarray(shape=(ns, nobs))

        for tt in range(nobs-1, -1, -1):
            if tt == (nobs-1):
                a_s[:, tt] = a_u[:, tt]
                Sa_s[:, :, tt] = Sa_u[:, :, tt]
            else:
                erru = reshape(a_s[:, tt+1]-a_p[:, tt+1], (ns, 1))
                errv = Sa_s[:, :, tt+1] - Sa_p[:, :, tt+1]
                iSa_p = inv(Sa_p[:, :, tt+1])
                SuTiSp = dot(dot(Sa_u[:, :, tt], transpose(T)), iSa_p)
                aux3 = reshape(dot(SuTiSp, erru), (ns,))
                aux4 = dot(dot(SuTiSp, errv), transpose(SuTiSp))
                a_s[:, tt] = a_u[:, tt] + aux3
                Sa_s[:, :, tt] = Sa_u[:, :, tt] + aux4

            a_s_std[:, tt] = sqrta(diag(Sa_s[:, :, tt]))

        mod['a_p'] = transpose(a_p)
        mod['a_u'] = transpose(a_u)
        mod['a_s'] = transpose(a_s)
        mod['Sa_p'] = Sa_p
        mod['Sa_u'] = Sa_u
        mod['Sa_s'] = Sa_s
        mod['a_s_std'] = transpose(a_s_std)
        mod['logLi'] = logLi
        mod['minuslogL'] = minuslogL.item()
        return mod