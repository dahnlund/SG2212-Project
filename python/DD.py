import numpy as np
import scipy.sparse as sp

def DD(n,h):
    """
    One-dimensional finite-difference derivative matrix 
    of size n times n for second derivative:
    h^2 * f''(x_j) = -f(x_j-1) + 2*f(x_j) - f(x_j+1)

    Homogeneous Neumann boundary conditions on the boundaries 
    are imposed, i.e.
    f(x_0) = f(x_1) 
    if the wall lies between x_0 and x_1. This gives then
    h^2 * f''(x_j) = + f(x_0) - 2*f(x_1) + f(x_2)
                   = + f(x_1) - 2*f(x_1) + f(x_2)
                   =              f(x_1) + f(x_2)

    For n=5 and h=1 the following result is obtained:
 
    A =
        -1     1     0     0     0
         1    -2     1     0     0
         0     1    -2     1     0
         0     0     1    -2     1
         0     0     0     1    -1
    """
    
    data = np.array([np.ones(n),-2*np.ones(n),np.ones(n)])
    A = sp.spdiags(data, [-1,0,1], format = 'csc')/h**2; A[0,0] = -1/h**2; A[-1,-1] = -1/h**2
    
    return A

if __name__ == "__main__":  
    print(DD(5,1).toarray())