import numpy as np

def avg(A,axis=0):
    """
    Averaging function to go from cell centres (pressure nodes)
    to cell corners (velocity nodes) and vice versa.
    avg acts on index idim; default is idim=1.
    """
    if (axis==0):
        B = (A[:,1:]+ A[:,:-1])/2
    elif (axis==1):
        B = (A[1:,:]+ A[:-1,:])/2
    else:
        raise ValueError('Wrong value for axis')
    return B           

if __name__ == "__main__":
    A = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]])
    print(avg(A,axis = 0))
    print()
    print(avg(A,axis = 1))