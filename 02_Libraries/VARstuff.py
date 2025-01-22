# VARstuff.py
# -----------
# Módulo que implementa algunas funciones útiles para el
# análisis de series de tiempo
#
# Escrito por Alan Ledesma - Enero 2020

# Funciones de librerías
from numpy import triu, diag, sqrt, asarray, arange
from numpy import zeros, ones, std, real, log, vstack
from numpy import eye, pi, hstack, concatenate, reshape
from numpy import kron, flipud, isinf, full, nan, dot
from numpy import sum as suma
from numpy.linalg import lstsq, inv, eigvalsh, cholesky
from scipy.special import gammaln
from numpy.random import normal, chisquare, randint





def get_yXform(data, lag, c=True):
    # Transforma la data en la representación MCO de un
    # VAR
    # USO: y, X, k, Tf = get_yXform(data, lag, cflag)
    # INPUTS:
    #   data [numpy array]: Matriz de orden (nobs x k) que
    #                       contiene la data
    #   lag [float]: número de rezagos del VAR
    #   cflag [Boolean]: Verdadero para añadir un intercepto
    #                  (default: True)
    # OUTPUT:
    #   y [numpy array]: Matriz de orden (nobs-lag x k) con
    #                    la data reorganizada
    #   X [numpy array]: Matriz de orden (nobs-lag x pk+c) con
    #                    la data reorganizada segun estructura
    #                    de rezagos
    #   k [float]: Número de variables incluidas
    #   Tf [float]: Número efectivo de observaciones
    #               (Tf=nobs-lag)
    #
    # Escrito por Alan Ledesma - Enero 2020
    T, k = data.shape
    y = data[lag:, :]
    X_slices = [data[(lag-ll-1):(T-ll-1), :] for ll in range(lag)]
    X = concatenate(X_slices, axis=1)
    if c:
        X = hstack((X, ones((T-lag,1))))

    return y, X, k, T-lag


def get_MCO_VAR(y, X, Bayes=False):
    # Estima el VAR por MCO
    # USO:
    #    Bhat, Sigmahat, yhat, uhat = get_MCO_VAR(y, X, BayesFlag)
    # INPUT:
    #   y [numpy array]: Matriz de orden (nobs-lag x k) con
    #                    la data reorganizada
    #   X [numpy array]: Matriz de orden (nobs-lag x p*k+c) con
    #                    la data reorganizada segun estructura
    #                    de rezagos
    #   BayesFlag [Boolean]: Verdadero para obtener calculos para
    #                        uso de procedimientos Bayesianos
    #                        (default: False)
    # OUTPUT:
    #   Bhat [numpy array]: Matriz de orden (p*k+c x k) que
    #                       contiene la estimación MCO de "B"
    #   Sigmahat [numpy array]: Matriz de orden (k x k) que
    #                           contiene la estimación MCO de
    #                           la covarianza de los residuos
    #                           (BayesFlag=False) o de la
    #                           covarianza scalada por grados
    #                           de libertad de los residuos
    #                           (BayesFlag=True).
    #   yhat [numpy array]: Matriz de orden (nobs-lag x k) con
    #                       la predicción de la data
    #   uhat [numpy array]: Matriz de orden (nobs-lag x k) con
    #                       el error de predicción de la data
    #
    # Escrito por Alan Ledesma - Enero 2020
    XX = dot(X.T,X)
    XY = dot(X.T,y)
    Bhat = lstsq(XX, XY, rcond=None)[0]
    yhat = dot(X,Bhat)
    uhat = y - yhat
    if Bayes:
        Sigmahat = dot(uhat.T,uhat)
    else:
        Tf, d = X.shape
        Sigmahat = dot(uhat.T,uhat)/(Tf-d)

    return Bhat, Sigmahat, yhat, uhat  # , AIC


def get_CompanionB(B):
    # Transforma la matriz "B" (de la representación MCO)
    # en su equivalente para la representación "Companion"
    # USO: BC = get_CompanionB(B)
    # INPUT: B [numpy array]: Matriz de orden (p*k+c x k) de
    #                         coeficientes
    # OUTPUT: BC [numpy array]: Matriz de orden (p*k+c x p*k+c)
    #                           de coeficientes en representación
    #                           companion
    #
    # Escrito por Alan Ledesma - Enero 2020
    ax, k = B.shape
    lag = ax // k
    BB = B[:lag*k, :].T
    below = hstack([eye((lag-1)*k), zeros(((lag-1)*k, k))])
    BB = vstack([BB, below])

    return BB


def get_IRF_Chol(Bhat, Sigmahat, h, shock, k, lag, scale=1, DoChol=True):
    # Obtiene una IRF
    # USO:
    #  IRF = get_IRF_Chol(B, S, h, s, k, p, scale, DoChol)
    # INPUT:
    #  B [numpy array]: Matriz de orden (p*k+c x k) de coeficientes
    #  S [numpy array]: Matriz de orden (k x k) que contiene la
    #                   covarianza de los residuos
    #  h [double]: Número de periodos en el IRF
    #  s [double]: Posición del shock en el vector de residuos
    #  k [double]: Número de variables en el VAR
    #  p [double]: Número de rezagos
    #  scale [double]: Número de desviaciones estándar del choque
    #                  (default: 1)
    #  DoChol [Boolean]: Verdadero para descomponer por cholesky
    #                    a "S" (default: True)
    # OUTPUT:
    #  IRF [numpy array]: Matriz de orden (h x k) que contiene la
    #                     respuesta de cada variable
    #
    # Escrito por Alan Ledesma - Enero 2020  
    if DoChol:
        C = cholesky(Sigmahat)
    else:
        C = Sigmahat

    #'''
    auxF = get_f(inv(C.T), C.T, Bhat, arange(1,h+2), k, lag)
    auxIRF = reordena(auxF, k, h+1)
    IRF = auxIRF[:,((shock-1)*k)+arange(k)].T*scale
    #'''
    '''
    u = zeros((k, 1))
    u[shock-1, 0] = scale
    BB = get_CompanionB(Bhat)
    MB = eye(k*lag)
    IRF0 = dot(C,u)
    IRF = IRF0

    for tt in range(1, h+1):
        MB = dot(MB,BB)
        IRF = hstack([IRF, dot(MB[:k, :k],IRF0)])
    '''

    return IRF


def get_IRF_Chol_BT(y, X, k, Tf, n, h, shock, lag, scale=1):
    # Obtiene varios IRF por bootstrap
    # USO: IRF = get_IRF_Chol_BT(y, X, k, Tf, n, h, s, p, scale)
    # INPUT:
    #  y [numpy array]: Matriz de orden (nobs-lag x k) con
    #                   la data reorganizada
    #  X [numpy array]: Matriz de orden (nobs-lag x p*k+c) con
    #                   la data reorganizada segun estructura
    #                   de rezagos
    #  k [double]: Número de variables en el VAR
    #  Tf [double]: Número de observaciones efectivas
    #  n [double]: Número de remuestreos
    #  h [double]: Número de periodos en el IRF
    #  s [double]: Posición del shock en el vector de residuos
    #  p [double]: Número de rezagos
    #  scale [double]: Número de desviaciones estándar del choque
    #                  (default: 1)
    # OUTPUT:
    #  IRF [numpy ndarray]: Matriz de orden (h x k x n) que contiene la
    #                       respuesta de cada variable por cada
    #                       remuestreo
    #
    # Escrito por Alan Ledesma - Enero 2020
    IRF_BT = zeros((k, h+1, n))
    Bhat, Sigmahat, _, uhat = get_MCO_VAR(y, X)
    IRF_BT[:, :, 0] = get_IRF_Chol(Bhat, Sigmahat, h, shock, k, lag, scale)

    step = n // 50
    print("[%s]\n[" % (" " * 50), end="", flush=True)
    for ii in range(1, n):
        X0sim = X[randint(0, Tf, 1), :]
        usim = uhat[randint(0, Tf, Tf), :]
        ysim, Xsim = get_sim_VAR(X0sim, usim, Bhat, lag)    
        Bsim, Sigmasim, _, _ = get_MCO_VAR(ysim, Xsim)
        IRF_BT[:, :, ii] = get_IRF_Chol(Bsim, Sigmasim, h, shock, k, lag, scale)
        
        if ii % step == 0:
            print("-", end="", flush=True)

    print("-]\n")
    return IRF_BT


def get_sim_VAR(X0, u, B, lag):
    # Simula una secuencia de variables endogenas
    # USO: ysim, Xsim = get_sim_VAR(X0, u, B, p)
    # INPUT:
    #  X0 [numpy array]: Matrix de orden (1 x kp+c)
    #                    punto incial para la simulación
    #  u [numpy array]: Matrix de orden (h x k) con "h"
    #                   residuos para la simulación
    #  B [numpy array]: Matriz de orden (p*k+c x k) de coeficientes
    #  p [double]: Número de rezagos
    # OUTPUT:
    # ysim [numpy array]: Matriz de orden (h x k) con varaibles
    #                     endógenas simuladas
    # Xsim [numpy array]: Matriz de orden (h x k*p+c) con varaibles
    #                     regresores MCO simulados
    #
    # Escrito por Alan Ledesma - Enero 2020
    T, k = u.shape
    X = zeros((T, k*lag+1))
    y = zeros((T, k))
    X[0, :] = reshape(X0, (1, k*lag+1))

    for tt in range(0, T):
        auxX = reshape(X[tt, :], (1, k*lag+1))
        y[tt, :] = dot(auxX,B) + u[tt, :]
        auxX = reshape(X[tt, 0:(lag-1)*k], (1, k*(lag-1)))
        auxy = reshape(y[tt, :], (1, k))
        if tt<T-1:
            X[tt+1, :] = hstack((auxy, auxX, ones(shape=(1, 1))))

    return y, X


def get_Fore_BT(y0, X0, uhist, B, lag, H, n):
    # Simula una secuencia de variables endogenas con remuestreo
    # USO: ysim = get_Fore_BT(y0, X0, uhist, B, p, h, n)
    # INPUT:
    #  y0 [numpy array]: Matrix de orden (1 x k) punto incial
    #  X0 [numpy array]: Matrix de orden (1 x kp+c)
    #                    punto incial para la simulación
    #  uhist [numpy array]: Matrix de orden (Tf x k) con residuos
    #                       en la muestra
    #  B [numpy array]: Matriz de orden (p*k+c x k) de coeficientes
    #  p [double]: Número de rezagos
    #  h [double]: Número de periodos a proyectar
    #  n [double]: Número de remuestreos
    # OUTPUT:
    #  ysim [numpy ndarray]: Matriz de orden (h x k x n) con varaibles
    #                        endógenas simuladas
    #
    # Escrito por Alan Ledesma - Enero 2020    
    Th, k = uhist.shape
    Fore_BT = zeros((k, H+1, n))

    for ii in range(n):
        Fore_BT[0, :, ii] = y0

    step = n // 50
    print("[%s]\n[" % (" " * 50), end="", flush=True)

    for ii in range(n):
        usim = uhist[randint(0, Th, H), :]
        ysim, _ = get_sim_VAR(X0, usim, B, lag)
        Fore_BT[1:(H+1), :, ii] = ysim

        if ii % step == 0:
            print("-", end="", flush=True)

    print("]\n")
    return Fore_BT

def Linf(A0, B, k, p):
    # Compute IRF for time horizon infinity
    # Ordering: kxk matrix which rows=response and column=shock

    BB = zeros((k, k))
    for ii in range(p):
        BB += B[slice(ii * k, (ii+1) * k), :]

    IBp = eye(k) - BB.T
    A0xIBp = dot(A0.T,IBp)
    Linfout = inv(A0xIBp)

    return Linfout

def Lh(A0, iA0, B, k, h, p):
    # Compute IRF for time horizon "1:h"
    # Ordering: kxkxh matrix which rows=response, column=shock, matrix=period

    Lhout = zeros((k, k, h))
    Lhout[:, :, 0] = iA0.T
    if h > 1:
        for tt in range(1, h):
            ef = min(tt, p)
            aux0 = zeros((k, k, ef))
            for ll in range(ef, 0, -1):
                bl = B[slice((ll - 1) * k, ll * k), :]
                M = dot(dot(iA0,bl),A0).T
                aux0[:, :, ll - 1] = dot(Lhout[:, :, tt  - ll],M)

            Lhout[:, :, tt] = suma(aux0, axis=2)
    
    
    return Lhout

def get_f(A0, iA0, B, hs, k, p):
    # Compute IRF for time horizon "1:h"
    # Ordering: (h*k)xk matrix which rows=response at period "row//k+1", column=shock
    infrest = isinf(hs)
    noinfhs = hs[~infrest]
    if len(noinfhs) > 0:
        aux = Lh(A0, iA0, B, k, max(noinfhs), p)

    rnir = len(noinfhs)
    aux = aux[:, :, noinfhs - 1]
    aux0 = full((rnir * k, k), nan)
    for ii in range(1, rnir + 1):
        inda = slice((ii - 1) * k, ii * k)
        aux0[inda, :] = aux[:, :, ii - 1]

    if any(infrest):
        aux1 = Linf(A0, B, k, p)
        fout = vstack((aux0, aux1))
    else:
        fout = aux0

    
    return fout

def reordena(fr, nk, np):
    # Reorganize IRF
    # Input: IRF with Ordering: (h*k)xk matrix which rows=response at period "row//k+1", column=shock
    # Output: IRF with Ordering: hx(k*k) matrix which rows=period, column=response of variable "column//k+1" of shock "column-column//k"
    fff = full((np, nk * nk), nan)
    vc = kron(ones((1, np)), arange(1, nk + 1))
    for ii in range(1, nk + 1):
        pos = reshape(vc == ii,(np*nk,))
        indf = arange(ii,(nk-1)*nk+1+ii,nk)-1
        fff[:,indf] = fr[pos, :]

    return fff


################################################
################################################
################################################
# Large BVAR
################################################
################################################
################################################


def get_DumObsLitterman(lmbda, deltas, sigmas, lag):
    #
    # Escrito por Alan Ledesma - Enero 2020

    k = sigmas.size
    sd = sigmas * deltas
    yd = diag(sd) / lmbda
    yd = vstack([yd, zeros(((lag-1)*k, k)),diag(sigmas), zeros((1, k))])

    Jp = diag(arange(1, lag+1))
    Xd = vstack([kron(Jp, diag(sigmas)/lmbda), zeros((k+1, lag*k))])
    LC = vstack([zeros(((lag+1)*k, 1)), asarray([1.e-3])])
    Xd = hstack([Xd, LC])


    Td, _ = yd.shape

    return yd, Xd, Td


def get_DumObsSumCoef(tau, deltas, mus, lag):
    #
    # Escrito por Alan Ledesma - Enero 2020

    k = deltas.size
    dm = deltas * mus
    yd = diag(dm) / tau

    Ip = ones((1, lag))
    Xd = hstack([kron(Ip, yd), zeros((k, 1))])

    Td, _ = yd.shape

    return yd, Xd, Td


def get_iXXchol(X):
    #
    # Escrito por Alan Ledesma - Enero 2020

    k, _ = X.shape
    J = flipud(eye(k))
    JXJ = dot(dot(J,X),J)
    Ct = cholesky(JXJ)
    iCt = inv(Ct).T
    iC = dot(dot(J,iCt),J)
    return iC

def chi2rnd(dof_vec):
    # generate a vector of independent draws from a Chi^2 distribution
    # Usage: chirnd_vec = chi2rnd(dof_vec)
    # Input:
    #     dof_vec [numpy array]: vector with degrees of freedoms
    # Output:
    #     chirnd_vec [numpy array]: vector which i-th elements is a draw from
    #                               the chi^2 distribution which dof is the
    #                               i-th element of dof_vec
    #
    # Written by Alan Ledesma - Jan 2020
    ch = suma(normal(0, 1, (dof_vec.size, max(dof_vec)))**2, axis=1)

    return ch


def LT_IWrnd(LT_S, dof):
    # generate one draw from a Inverse-Wishart distribution
    # Usage: LT_S_out = LT_IWrnd(LT_S_in,dof)
    # Input:
    #     LT_S_in [numpy array]: Cholesky decomposition (Lower trinagular) of
    #                            matrix Sigma (such that LT_S*LT_S'=Sigma)
    #     dof [Scalar]: Degrees of freedoms
    # Output:
    #     LT_S_out [numpy array]: Cholesky decomposition (Lower trinagular)
    #                             of drew matrix Sigma
    #                             (such that LT_S*LT_S'=Sigma)
    #
    # Written by Alan Ledesma - Jan 2020
    c, _ = LT_S.shape
    z = triu(normal(0, 1, (c, c)), 1)
    QT = diag(sqrt(chisquare(dof - c + arange(1, c + 1))))
    QT += z
    LT_S_draw = lstsq(QT, LT_S.T, rcond=None)[0].T
    return LT_S_draw


def MatNormrnd(Mu, LT_P, LT_S):
    # generate one draw from a Matricvariate Normal distribution
    # Usage: x_out = MatNormrnd(Mu,LT_S,LT_Psi)
    # Input:
    #     Mu [numpy array]: Mean matrix, order=(m,n)
    #     LT_P [numpy array]: Cholesky decomposition (Lower trinagular) of
    #                         right scaling matrix (such that LT_P*LT_P'=Psi)
    #                         order=(m,m)
    #     LT_S [numpy array]: Cholesky decomposition (Lower trinagular) of
    #                         left scaling matrix (such that LT_S*LT_S'=Sigma)
    #                         order=(n,n)
    # Output:
    #     x_out [numpy array]: drew matrix, order=(m,n)
    #
    # Written by Alan Ledesma - Jan 2020
    z = normal(0, 1, Mu.shape)
    x_draw = Mu + dot(dot(LT_P,z),LT_S.T)

    return x_draw


def get_BVARforvast(y0, X0, lag, h, b, r, Bp, LT_SSp, LT_Pp, Dofp):
    #
    # Escrito por Alan Ledesma - Enero 2020
    k = y0.size
    FF = zeros((h+1, k, b+r))
    for rr in range(b+r):
        FF[0, :, rr] = y0

    step = (r + b) // 50
    print("[%s]\n[" % (" " * 50), end="", flush=True)

    for rr in range(b+r):
        # Draw coefs
        LT_S_draw = LT_IWrnd(LT_SSp, Dofp)
        B_draw = MatNormrnd(Bp, LT_Pp, LT_S_draw)
        usim = dot(normal(0, 1, (h, k)),LT_S_draw)
        # get 1 forecast
        ysim, _ = get_sim_VAR(X0, usim, B_draw, lag)
        FF[1:(h+1), :, rr] = ysim

        if rr % step == 0:
            print("-", end="", flush=True)

    print("]\n")
    FF = FF[:, :, b:b+r]
    return FF


def get_IRF_Chol_BVAR(lag, h, b, r, Bp, LT_SSp, LT_Pp, Dofp, shock, scale):
    #
    # Escrito por Alan Ledesma - Enero 2020
    k, _ = LT_SSp.shape
    IRF = zeros((k, h+1, r))

    step = (r + b) // 50
    print("[%s]\n[" % (" " * 50), end="", flush=True)

    for ii in range(r+b):
        # Draw coefs
        LT_S_draw = LT_IWrnd(LT_SSp, Dofp)
        B_draw = MatNormrnd(Bp, LT_Pp, LT_S_draw)
        # Draw IRF: Sigma = LT_S_draw*LT_S_draw'
        if ii>=b:
            IRF[:, :, ii-b] = get_IRF_Chol(B_draw, LT_S_draw, h, shock, k, lag, scale, False)

        if ii % step == 0:
            print("-", end="", flush=True)

    print("-]\n")
    return IRF


def get_sigmas(data, lag):
    #
    # Escrito por Alan Ledesma - Enero 2020
    T, k = data.shape
    sigmas = zeros((k))
    for ii in range(k):
        XX = ones((T-lag, lag+1))
        YY = data[lag:, ii]

        for ll in range(lag):
            XX[:, ll+1] = data[lag-ll-1:T-ll-1, ii]

        B = lstsq(dot(XX.T,XX), dot(XX.T,YY), rcond=None)[0]
        UU = YY - dot(XX,B)
        sigmas[ii] = std(UU)
    return sigmas

def Hyper_ML(theta, X, y, sigmas, deltas, flag_sc, mus):
    T,k  = y.shape
    n,klagp1 = X.shape
    lag = (klagp1-1)//k
    lmbda = theta[0]
    
    yd, Xd, Td = get_DumObsLitterman(lmbda, deltas, sigmas, lag)
    
    if flag_sc = True:
        tau = theta[1]
        yc, Xc, Tc = get_DumObsSumCoef(tau, deltas, mus, lag)
        yd = vstack((yd, yc))
        Xd = vstack((Xd, Xc))
        Td = Td + Tc
    
    Text = T + Td
    
    constants = -((k*T)/2)*log(pi) + suma( gammaln( (Text-n+1-(1+arange(k)))/2) ) - suma( gammaln( (Td-n+1-(1+arange(k)))/2) )
    Omegaprior = diag(1/diag(dot(Xd.T,Xd)))
    Uprior = dot((eye(Td)-dot(Xd,dot(Omegaprior,Xd.T))),yd)
    Sprior = diag(diag(dot(Uprior.T,Uprior)))
    Sprior_logdet = suma(log(diag(Sprior)))
    yext = vstack((yd, y))
    Xext = vstack((Xd, X))
    iOmegapost = dot(Xext.T,Xext)
    Omegapost = inv(iOmegapost)
    Upost      = dot((eye(Text)-dot(Xext,dot(Omegapost,Xext.T))),yext)
    Spost      = dot(Upost.T,Upost)
    
    LiS     = diag(1./sqrt(diag(Sprior)))
    LOmega  = cholesky(Omegaprior)
    M       = Spost-Sprior
    LXL     = dot(LOmega.T,dot(dot(X.T,X),LOmega))
    LML     = dot(LiS.T,dot(M,LiS))
    eigLXL  = real(eigvalsh(LXL))
    eigLML  = real(eigvalsh(LML))
    LXLp1_detlog = suma(log(1+eigLXL))
    LMLp1_detlog = suma(log(1+eigLML))
    
    resto   = -(T/2)*Sprior_logdet - (k/2)*LXLp1_detlog - ((Text-n)/2)*LMLp1_detlog
        
    logML = constants + resto

    return logML
