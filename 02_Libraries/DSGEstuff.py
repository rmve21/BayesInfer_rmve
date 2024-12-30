from numpy import asarray, dot, transpose, diag, c_, real, zeros, ndarray, reshape
from numpy import identity, fill_diagonal, finfo, all, spacing, r_
from numpy.linalg import matrix_rank, inv, svd, eigvals
from math import sqrt, exp, log
from scipy.linalg import qz
from scipy.linalg import solve_discrete_lyapunov as lyapunov
from scipy.special import beta, gamma
from KFstuff import KFplus
from GeneralComplements import nearestSPD, matprint

########################################
##### Functions to solve the DSGE
########################################
def qzswitch(i, A, B, Q, Z):
    #  translation of qzswitch by Chris Sims
    # Takes U.T. matrices A, B, orthonormal matrices Q,Z, interchanges
    # diagonal elements i and i+1 of both A and B, while maintaining
    # Q'AZ' and Q'BZ' unchanged.  Does nothing if ratios of diagonal elements
    # in A and B at i and i+1 are the same.  Aborts if diagonal elements of
    # both A and B are zero at either position.
    # Translated by Sarunas Girdenas and Hyun Changi Yi
    a, d, b, e, c, f = A[i-1][i-1], B[i-1][i-1], A[i-1][i], B[i-1][i], A[i][i], B[i][i]
    wz = asarray([c*e - f*b, c*d - f*a])
    xy = asarray([b*d - e*a, c*d - f*a])
    n = sqrt(dot(wz, wz))
    m = sqrt(dot(xy, xy))
    if n == 0:
        return A, B, Q, Z
    else:
        wz = wz/n # the original code uses inverse matrix division '\' like n\wz
        xy = xy/m
        wz = [wz, [-wz[1], wz[0]]]
        xy = [xy, [-xy[1], xy[0]]]
        # replace (j+1)th and (j+2)th rows of A by matrix multiplication of xy and itself
        A[[i-1,i], :] = dot(xy, A[[i-1,i], :])
        B[[i-1,i], :] = dot(xy, B[[i-1,i], :])
        A[:, [i-1,i]] = dot(A[:, [i-1,i]], wz)
        B[:, [i-1,i]] = dot(B[:, [i-1,i]], wz)
        Z[:, [i-1,i]] = dot(Z[:, [i-1,i]], wz)
        Q[[i-1,i], :] = dot(xy, Q[[i-1,i], :])
    return A, B, Q, Z

def qzdiv(stake, A, B, Q, Z,flagEstima):
    # translation of qzdiv by Chris Sims
    # Takes U.T. matrices A, B, orthonormal matrices Q,Z, rearranges them
    # so that all cases of abs(B(i,i)/A(i,i))>stake are in lower right 
    # corner, while preserving U.T. and orthonormal properties and Q'AZ' and
    # Q'BZ'.
    # Translated by Alan Ledesma

    n = A.shape[0]
    root = transpose(asarray([abs(diag(A)), abs(diag(B))]))
    root[:,0] = root[:,0]-(root[:,0]<1e-13)*(root[:,0]+root[:,1])
    root[:,1] = root[:,1]/root[:,0]
    for i in range(n-1, -1, -1):
        m = -1
        for j in range(i, -1, -1):
            if (root[j,1] > stake or root[j,1] < -.1):
                m = j
                break
                
        if m == -1:
            return A, B, Q, Z
        
        if m == i:
            if not flagEstima:
                print('m is equal to i')
            
        for k in range(m, i):
            A,B,Q,Z = qzswitch(k+1, A, B, Q, Z)
            tmp = root[k,1]
            root[k,1] = root[k+1,1]
            root[k+1,1] = tmp
            
        
    return A,B,Q,Z

def SolveDSGE(modelo,flagEstima=False):

    fyp = modelo['SLER']['fyp']
    fxp = modelo['SLER']['fxp']
    fy  = modelo['SLER']['fy'] 
    fx  = modelo['SLER']['fx'] 
    Lambda = modelo['SLER']['Lambda']
    Eta = modelo['SLER']['Eta']

    n,ny = fy.shape
    nx   = fx.shape[1]
    nx2  = Lambda.shape[1]
    nx1  = nx-nx2
    ns   = Eta.shape[1]
    dims = {'n':n,'ny':ny,'nx1':nx1,'nx2':nx2,'nx':nx,'ns':ns}
    modelo['SLER']['dimensiones'] = dims 

    A = c_[-fxp,-fyp]
    B = c_[fx,fy]
    NK = fx.shape[1] # Número de variables predeterminadas
    s, t, q, z = qz(-A, B)
    stake = 1 # Es estándar mantenerlo en "1"
    slt = abs(diag(t)) < stake*abs(diag(s)) # Raices estables
    nk = sum(slt) # Número de raices estables
    # Ordenando la descompsición QZ agrupando por raices estables
    s, t, q, z = qzdiv(stake, s, t, q, z,flagEstima)
    # Agrupando elementos
    z21 =  z[nk:, 0:nk]
    z11 =  z[0:nk,0:nk]
    s11 = -s[0:nk, 0:nk] 
    t11 =  t[0:nk, 0:nk]
    # Condiciones de estabilidad
    Estable = True
    if nk > NK:
        if not flagEstima:
            print('El equilibrio esta localmente indeterminado')

        Estable = False
        
    elif nk < NK:
        if not flagEstima:
            print('No existe equilibrio local')

        Estable = False

    if Estable:
        if matrix_rank(z11) < nk:
            if not flagEstima:
                print('Invertibility condition violated')

            Estable = False

    if Estable:
        z11i  = inv(z11)
        s11i  = inv(s11)
        gx = real(dot(z21,z11i))
        hx = real(dot(z11,dot(dot(s11i,t11),z11i)))
        # To gain accuracy
        hx[nx1:,0] = 0
        hx[nx1:,nx1:] = Lambda
    else:
        gx = ndarray(shape=(ny,nx))
        hx = ndarray(shape=(nx,nx))
            
    solucion = {'gx':gx,'hx':hx,'Eta':Eta,'Estable':Estable}
    modelo['solución'] = solucion

    return modelo

def genIRF(solvedmodel,nper=25):
    gx = solvedmodel['solución']['gx']
    hx = solvedmodel['solución']['hx']
    Eta = solvedmodel['solución']['Eta']
    ny = solvedmodel['SLER']['dimensiones']['ny']
    nx1 = solvedmodel['SLER']['dimensiones']['nx1']
    nx2 = solvedmodel['SLER']['dimensiones']['nx2']
    ns = solvedmodel['SLER']['dimensiones']['ns']
    
    Y = ndarray(shape=(ny,nper,ns))
    X = ndarray(shape=(nx1+nx2,nper,ns))
    for ss in range(ns):
        shock = zeros(shape=(ns,1))
        shock[ss,0] = 1
        for tt in range(nper):
            if tt == 0:
                x = dot(Eta,shock)
            else:
                x = dot(hx,x)
            
            y = dot(gx,x)
            Y[:,tt,ss] = reshape(y,(ny,))
            X[:,tt,ss] = reshape(x,(nx1+nx2,))
        
    return Y,X

########################################
##### Functions to filter
########################################
def get_StatesMat(solvedmodel,stationaryflag=True):
    # Unpack dimensions
    n = solvedmodel['SLER']['dimensiones']['n']
    nx = solvedmodel['SLER']['dimensiones']['nx']
    nx1 = solvedmodel['SLER']['dimensiones']['nx1']
    nx2 = solvedmodel['SLER']['dimensiones']['nx2']
    ny = solvedmodel['SLER']['dimensiones']['ny']
    ns = solvedmodel['SLER']['dimensiones']['ns']
    hx = solvedmodel['solución']['hx']
    gx = solvedmodel['solución']['gx']
    Eta = solvedmodel['SLER']['Eta']
    Lambda = solvedmodel['SLER']['Lambda']
    # Unpack solution
    g1x = reshape(gx[:,:nx1],(ny,nx1))
    g2x = reshape(gx[:,nx1:],(ny,nx2))
    Etatilde = reshape(Eta[nx1:,:],(nx2,nx2))
    h1x = reshape(hx[:nx1,:nx1],(nx1,nx1))
    h2x = reshape(hx[:nx1,nx1:],(nx1,nx2))
    # State system matrices
    c = zeros(shape=(n,1))
    Tup = c_[zeros(shape=(ny,ny)),g1x,dot(g2x,Lambda)]
    Tmd = c_[zeros(shape=(nx1,ny)),h1x,dot(h2x,Lambda)]
    Tdw = c_[zeros(shape=(nx2,ny+nx1)),Lambda]
    T   = r_[Tup,Tmd,Tdw]
    R   = dot(r_[g2x,h2x,identity(nx2)],Etatilde)
    Q   = identity(ns)
    if stationaryflag:
        # alpha0 = np.dot(np.linalg.inv(np.identity(n)-T),c)
        # pero c=0
        alpha0 = c
        # Solve: AXA' - X + Q = 0
        AA = T
        QQ = dot(dot(R,Q),R.T)
        # In case it is no SPD
        Sigma_a0 = lyapunov(AA,QQ,method='bilinear')
        Sigma_a0 = nearestSPD(Sigma_a0)
    else:
        alpha0 = c
        Sigma_a0 = identity(n)*1000#00000
        
    
    mod = {'c':c,'T':T,'R':R,'Q':Q,'alpha0':alpha0,'Sigma_a0':Sigma_a0}
    solvedmodel['REE'] = mod        
    return solvedmodel

########################################
##### Functions to estimate
########################################
def get_minusloglike(theta,data,SLERfun,Measurefun,flagEstima=True):
    modelo = dict()
    modelo['SLER'] = SLERfun(theta) # Sistema lineal de expectativas racionales
    modelo = SolveDSGE(modelo) # Solución
    modelo = get_StatesMat(modelo,True) # Sistema de ecuaciones de estado
    modelo = Measurefun(modelo) # Sistema de ecuaciones de espacio
    ops = {'compute_stderrors':False,'a_initial':modelo['REE']['alpha0'].flatten(),'P_initial':modelo['REE']['Sigma_a0'],'only_likelihood':flagEstima}
    aux0 = KFplus(data,modelo['REE'],ops)
    if flagEstima:
        return aux0
    else:
        return aux0['minuslogL'],aux0


def get_minuslogpost(theta,data,SLERfun,Measurefun,minuslogpriorfun,flagEstima=True):
    minuslogprior = minuslogpriorfun(theta)
    if flagEstima:
        minusloglike = get_minusloglike(theta,data,SLERfun,Measurefun,flagEstima)
        return minuslogprior+minusloglike
    else:
        minusloglike,modfiltered = get_minusloglike(theta,data,SLERfun,Measurefun,flagEstima)
        return minuslogprior+minusloglike,modfiltered

def get_MHsim(n, b, x0, LTOmega, scale, data,SLERfun,Measurefun,minuslogpriorfun):
    from numpy import zeros, reshape, log, dot, ndarray, delete, isnan
    from numpy.random import uniform, randn
    from sys import stdout
    from math import floor

    k = LTOmega.shape[0]
    xsim = zeros(shape=(k, n+b))
    aceptance = 1

    toolbar_width = 50
    step = max(1, int((n + b) / toolbar_width))
    print("%s]\n[" % (" " * toolbar_width), end="", flush=True)
    for ii in range(n+b):
        if ii == 0:
            xsim[:, ii] = reshape(x0, (k,))
            x_old = reshape(xsim[:, ii], (k,))  
            fx_old = -1*get_minuslogpost(x_old,data,SLERfun,Measurefun,minuslogpriorfun)
        else:
            eu = uniform(0, 1)
            if eu < 1e-20:
                u = -10**10
            else:
                u = log(eu)

            x_new = x_old + reshape(scale*dot(LTOmega, randn(k, 1)), (k,)) # Candidate
            mfx_new,aux0 = get_minuslogpost(x_new,data,SLERfun,Measurefun,minuslogpriorfun,False)
            fx_new = -mfx_new
            alpha = min(fx_new-fx_old, 0)

            if ii == b:
                a_s = aux0['a_s']
                Tt, nv = a_s.shape
                smooth_sim = ndarray(shape=(Tt, nv, n))
                updated_sim = ndarray(shape=(Tt, nv, n))
                predicted_sim = ndarray(shape=(Tt, nv, n))

            a_s = aux0['a_s']
            a_u = aux0['a_u']
            a_p = aux0['a_p']
            if u <= alpha and not isnan(x_new).any() and not isnan(a_s).any():
                fx_old = fx_new
                x_old = x_new
                xsim[:, ii] = x_new
                aceptance += 1
                if ii >= b:
                    smooth_sim[:, :, ii-b] = a_s
                    updated_sim[:, :, ii-b] = a_u
                    predicted_sim[:, :, ii-b] = a_p
            else:
                xsim[:, ii] = xsim[:, ii-1]
                if ii >= b:
                    smooth_sim[:, :, ii-b] = smooth_sim[:, :, ii-b-1]
                    updated_sim[:, :, ii-b] = updated_sim[:, :, ii-b-1]
                    predicted_sim[:, :, ii-b] = predicted_sim[:, :, ii-b-1]

        if (ii % step) == 0:
            print(f"\r{'-'*(ii//step)} R. Acc. = {(aceptance/(ii+1))*100:.2f}", end="")


    print(f"\r{'-'*(ii//step)}]")
    xsim = delete(xsim, range(b), 1)
    return xsim, (aceptance/(n+b))*100, smooth_sim, updated_sim, predicted_sim

########################################
##### Some PDFs
########################################
def BetaPDF(x,mu,sd):
    # Usage: p = Beta_PDF(x,mu,sd)
    # Beta PDF where "a" and "b" are such that "mu" and "sd" are the
    # inconditional mean and standard deviation
    # Alan Ledesma, Nov-2015.    
    eps = finfo(float).eps
    v = sd**2
    a = ( (mu**2)*(1-mu) - mu*v )/v
    b = ( mu*((1-mu)**2) - (1-mu)*v )/v
    if x<0 or x>1:
        return 0
    elif x==0:
        if a>1:
            return 0
        
        if a<1:
            return 1/eps

    elif x==1:
        if b>1:
            return 0

        if b<1:
            return 1/eps
    
    else:
        return ( (x**(a-1))*((1-x)**(b-1)) )/beta(a,b)

def InvGammaPDF(x,mu,sd):
    # Usage: p = InvGammaPDF(x,mu,sd)
    # inverse Gamma PDF where "a" and "b" are such that "mu" and "sd" are the
    # inconditional mean and standard deviation
    # Alan Ledesma, Nov-2015.
    if x<=0:
        return 0
    else:
        v = sd**2
        a = (mu**2)/v + 2
        b = mu*( (mu**2)/v + 1 )
        return ( (b**a)/gamma(a) )*x**(-a-1)*exp(-b/x)




########################################
##### Other usefull functions
########################################

    

def printREE(modREE):
    print('---- Matriz de transición T:')
    matprint(modREE['T'])
    print('---- Matriz R:')
    matprint(modREE['R'])
    print('---- Matriz de carga Z:')
    matprint(modREE['Z'])
    print('---- Matriz varianza H:')
    matprint(modREE['H'])


## MH para DSGE
def get_MHdraws(n, b, SLERfun,Measurefun,minuslogpriorfun, x0, LTOmega, scale, data):
    from numpy import zeros, reshape, log, dot, ndarray, delete, isnan
    from numpy.random import uniform, randn
    from sys import stdout
    from math import floor

    k = LTOmega.shape[0]
    T = data.shape[0]
    xsim = zeros(shape=(k, n+b))
    aceptance = 1

    toolbar_width = 50
    step = max(1, int((n + b) / toolbar_width))
    print("%s]\n[" % (" " * toolbar_width), end="", flush=True)
    for ii in range(n+b):
        if ii == 0:
            xsim[:, ii] = reshape(x0, (k,))
            x_old = reshape(xsim[:, ii], (k,))
            fx_old = -get_minuslogpost(x_old,data,SLERfun,Measurefun,minuslogpriorfun,True)
        else:
            eu = uniform(0, 1)
            if eu < 1e-20:
                u = -10**10
            else:
                u = log(eu)

            Estable = False
            Regular = False
            while not Regular:
                while not Estable:
                    x_new = x_old + reshape(scale*dot(LTOmega, randn(k, 1)), (k,)) # Candidate
                    modelo = dict()
                    modelo['SLER'] = SLERfun(x_new) # Sistema lineal de expectativas racionales
                    modelo = SolveDSGE(modelo,True) # Solución
                    Estable = modelo['solución']['Estable']

                modelo = get_StatesMat(modelo,True) # Sistema de ecuaciones de estado
                modelo = Measurefun(modelo) # Sistema de ecuaciones de espacio
                ops = {'compute_stderrors':False,'a_initial':modelo['REE']['alpha0'].flatten(),'P_initial':modelo['REE']['Sigma_a0'],'only_likelihood':False}
                modfiltered = KFplus(data,modelo['REE'],ops)
                
                a_s = modfiltered['a_s']
                a_p = modfiltered['a_p']
                a_u = modfiltered['a_u']
                Regular = not isnan(x_new).any() 
                Regular = Regular and not isnan(a_p).any()
                Regular = Regular and not isnan(a_u).any()
                Regular = Regular and not isnan(a_s).any()

            if ii==1:
                ns = a_s.shape[1]
                smooth_sim = ndarray(shape=(T, ns, n))
                updated_sim = ndarray(shape=(T, ns, n))
                predicted_sim = ndarray(shape=(T, ns, n))
                
            fx_new = -minuslogpriorfun(x_new)-modfiltered['minuslogL']
            alpha = min(fx_new-fx_old, 0)
            if u <= alpha:
                fx_old = fx_new
                x_old = x_new
                xsim[:, ii] = x_new
                aceptance += 1
                if ii >= b:
                    smooth_sim[:, :, ii-b] = a_s
                    updated_sim[:, :, ii-b] = a_u
                    predicted_sim[:, :, ii-b] = a_p[1:,:]
            else:
                xsim[:, ii] = xsim[:, ii-1]
                if ii >= b:
                    smooth_sim[:, :, ii-b] = smooth_sim[:, :, ii-b-1]
                    updated_sim[:, :, ii-b] = updated_sim[:, :, ii-b-1]
                    predicted_sim[:, :, ii-b] = predicted_sim[:, :, ii-b-1]

        if (ii % step) == 0:
            print(f"\r{'-'*(ii//step)} R. Acc. = {(aceptance/(ii+1))*100:.2f}%", end="")
  

    print(f"\r{'-'*(ii//step)}]")
    xsim = delete(xsim, range(b), 1)
    return xsim, (aceptance/(n+b))*100, smooth_sim, updated_sim, predicted_sim
