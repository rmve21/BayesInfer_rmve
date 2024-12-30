from numpy import dot, real, spacing, eye, zeros
from numpy import diag, c_
from numpy.linalg import inv, svd, eigvals
from numpy import finfo

# Printing nice fommated matrices
def matprint(mat, fmt="g"):
    nd = mat.ndim
    if nd == 1:
        mat = mat.reshape(1, -1)

    col_widths = [max(map(lambda x: len(("{:"+fmt+"}").format(x)), col)) for col in mat.T]
    
    for row in mat:
        print("  ".join(("{:"+fmt+"}").format(value) for value, width in zip(row, col_widths)))


def nearestSPD(A):
    # nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
    # usage: Ahat = nearestSPD(A)
    # From Higham: "The nearest symmetric positive semidefinite matrix in the
    # Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
    # where H is the symmetric polar factor of B=(A + A')/2."
    #
    # http://www.sciencedirect.com/science/article/pii/0024379588902236
    #
    # arguments: (input)
    #  A - square matrix, which will be converted to the nearest Symmetric
    #    Positive Definite Matrix.
    #
    # Arguments: (output)
    #  Ahat - The matrix chosen as the nearest SPD matrix to A.
    #
    # Written by Alan Ledesma
    
    # test for a square matrix A
    r,c = A.shape
    if r != c:
        raise ValueError('"A" must be a square matrix.')
    elif r == 1 and A <= 0:
        print('"A" is non-positive scalar, so I am just returning an eps')
        return finfo(float).eps
    
    # symmetrize A into B
    B = (A + A.T)/2
    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    _,Sigma,V = fullsvd(B)
    H = dot(dot(V,Sigma),V.T)
    # get Ahat
    Ahat = (B+H)/2
    # ensure symmetry
    Ahat = (Ahat + Ahat.T)/2
    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    p = True
    k = 0
    while p != 0:
        p = not all(eigvals(Ahat) > 0)
        k = k + 1
        if p:
            # Ahat failed being PD. It must have been just a hair off,
            # due to floating point trash, so it is simplest now just to
            # tweak by adding a tiny multiple of an identity matrix.
            mineig = real(min(eigvals(Ahat)))
            Ahat = Ahat + (-mineig*k**2 + spacing(mineig))*eye(A.shape[0])        
        

    return Ahat

def fullsvd(A):
    # compute SVD decomposition with Matlab type outcomes
    U, sdiag, VH = svd(A, full_matrices=False)
    S = diag(sdiag)
    V = VH.T.conj()
    return U,S,V


def HP(Y,Lambda):
    # HP trend and cycle
    # This function estimates the trend and Cycle using the 
    # Hodrick & Prescott filter published in
    #
    # Hodrick, R. J., and Prescott, E. C., “Post-war U.S. 
    #      business cycles: An empirical investigation.” 
    #      Discussion paper 451, Carnegie-Mellon University,1980.
    #
    # SINTAX: Trend,cycle = HP(Y,Lambda)
    # INPUT: 
    #   Y [vector]: Time series data for the variable to filter.
    #   Lambda [scalar]: HP weights
    # OUPUT:
    #   Trend [vector]: HP trend
    #   Cycle [vector]: HP cycle
    #
    # Written by Alan Ledesma
    T = Y.shape[0]
    A = eye(T-2)
    B1 = c_[A,zeros(shape=(T-2,2))]
    B2 = c_[zeros(shape=(T-2,1)),-2*A,zeros(shape=(T-2,1))]
    B3 = c_[zeros(shape=(T-2,2)),A]
    B  = B1+B2+B3
    F  = dot(B.T,B)
    Trend = dot(inv(Lambda*F+eye(T)),Y)
    Cycle = dot(Lambda*F,Trend)


    return Trend,Cycle