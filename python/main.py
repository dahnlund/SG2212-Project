#%%
'''Load dependencies'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.animation
import math 
import scipy.sparse as sp
import scipy.linalg as scl
from scipy.sparse.linalg import splu
params = {'legend.fontsize': 12,
          'legend.loc':'best',
          'figure.figsize': (8,5),
          'lines.markerfacecolor':'none',
          'axes.labelsize': 12,
          'axes.titlesize': 12,
          'xtick.labelsize':12,
          'ytick.labelsize':12,
          'grid.alpha':0.6}
pylab.rcParams.update(params)

from DD import DD
from avg import avg
from src import tic, toc

#%% Simulation parameters

Pr = 0.71
Re = 25
# Ri = 0. 
dt = 0.001
Tf = 5
Lx = 1.
Ly = 1.
Nx = 20
Ny = 20
namp = 0.
ig = 20

#%% Discretization in space and time, and definition of boundary conditions

# number of iteratins
Nit = int(Tf/dt)
# edge coordinates
x = np.linspace(0,Lx,Nx+1)
y = np.linspace(0,Ly, Ny+1)
# grid spacing
hx = x[-1]/(Nx)
hy = y[-1]/(Ny)

# boundary conditions
Utop = 1; Ttop = 1.; Tbottom = 0.;
uN = x*0 + Utop;  uN = uN[:,np.newaxis];    vN = avg(x)*0;    vN = vN[:,np.newaxis];
uS = x*0;  uS = uS[:,np.newaxis];         vS = avg(x)*0;  vS = vS[:,np.newaxis];
uW = avg(y)*0;  uW = uW[np.newaxis,:];       vW = y*0;  vW = vW[np.newaxis,:];
uE = avg(y)*0;  uE = uE[np.newaxis,:];       vE = y*0;    vE = vE[np.newaxis,:];

tN = 100; tS = 10

#%% Pressure correction and pressure Poisson equation

# Compute system matrices for pressure 
# Laplace operator on cell centres: Fxx + Fyy
# First set homogeneous Neumann condition all around
#Lp = np.kron(....).toarray(),DD(.....).toarray()) + np.kron(.....).toarray(),sp.eye(.....).toarray());
Lp = sp.kron(sp.eye(Ny, format = 'csc'), DD(Nx,hx), format = 'csc') \
    + sp.kron(DD(Ny,hy), sp.eye(Nx, format = 'csc'), format = 'csc')
# Set one Dirichlet value to fix pressure in that point
#Lp[:,0] = 0; Lp[0,:] =0; Lp[0,0] = 1;
Lp[0,:] =0; Lp[0,0] = 1;
Lps_lu = splu(Lp)

#%% Initial conditions

U = np.zeros((Nx-1,Ny))
V = np.zeros((Nx,Ny-1))

T = 0.5 + \
    namp*(np.random.rand(Nx,Ny)-0.5); 

#%% Main time-integration loop. Write output file "cavity.mp" if

if (ig>0):
    metadata = dict(title='Lid-driven cavity', artist='SG2212')
    writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=metadata)
    matplotlib.use("Agg")
    fig=plt.figure()
    writer.setup(fig,"cavity.mp4",dpi=200)

# progress bar
print('[         |         |         |         |         ]')
tic()
for k in range(Nit):
    # print("Iteration k=%i time=%.2e" % (k,k*dt))

    # include all boundary points for u and v (linear extrapolation
    # for ghost cells) into extended array (Ue,Ve)
    Ue = np.vstack((uW, U, uE)); Ue = np.hstack( (2*uS-Ue[:,0,np.newaxis], Ue,  2*uN-Ue[:,-1,np.newaxis]))
    Ve = np.hstack((vS, V, vN)); Ve = np.vstack( (2*vW-Ve[0,:,np.newaxis].T, Ve, 2*vE-Ve[-1,:,np.newaxis].T));

    # averaged (Ua,Va) of u and v on corners
    Ua = avg(Ue, axis = 1)
    Va = avg(Ve, axis = 0) 

    #  construct individual parts of nonlinear terms
    dUVdx = np.diff( Ua*Va, axis=0)/hx;
    dUVdy = np.diff( Ua*Va, axis=1)/hy;
    Ub    = avg( Ue[:,1:-1],0);   
    Vb    = avg( Ve[1:-1,:],1);
    dU2dx = np.diff( Ub**2, axis = 0 )/hx;
    dV2dy = np.diff( Vb**2, axis = 1 )/hy;

    # treat viscosity explicitly
    viscu = np.diff( Ue[:,1:-1],axis=0,n=2 )/hx**2 + \
         np.diff( Ue[1:-1,:],axis=1,n=2 )/hy**2;
    viscv = np.diff( Ve[:,1:-1],axis=0,n=2 )/hx**2 + \
         np.diff( Ve[1:-1,:],axis=1,n=2 )/hy**2;

    # compose final nonlinear term + explicit viscous terms
    U = U + dt*( -dU2dx -dUVdy[1:-1,:] + viscu/Re)
    V = V + dt*( -dUVdx[:,1:-1] - dV2dy + viscv/Re)

    # pressure correction, Dirichlet P=0 at (1,1)
    rhs = (np.diff(np.vstack((uW, U, uE)), axis=0)/hx + np.diff(np.hstack((vS, V, vN)),axis=1)/hy)/dt;
    rhs = np.reshape(rhs.T,(Nx*Ny,1));
    rhs[0] = 0;

    # different ways of solving the pressure-Poisson equation:
    P = Lps_lu.solve(rhs)
    #P = sp.linalg.spsolve(Lp, rhs, use_umfpack = False)
    P = np.reshape(P.T, (Ny,Nx)).T

    # apply pressure correction
    U = U - dt*np.diff(P, axis = 0)/hx;
    V = V - dt*np.diff(P, axis = 1)/hy; 

    # Temperature equation
    #....
    """
    # do postprocessing to file
    if (ig>0 and np.floor(k/ig)==k/ig):
        plt.clf()
        plt.contourf(avg(x),avg(y),T.T,levels=np.arange(0,1.05,0.05))
        plt.gca().set_aspect(1.)
        plt.colorbar()
        plt.title(f'Temperature at t={k*dt:.2f}')
        writer.grab_frame()

    # update progress bar
    if np.floor(51*k/Nit)>np.floor(51*(k-1)/Nit): 
        print('.',end='')
    """
    if (ig>0 and np.floor(k/ig)==k/ig):
        Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
        Va = np.vstack((vW,avg(np.hstack((vS,V,
                                        vN)),0),vE));
        plt.clf()
        plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20)
        plt.quiver(x,y,Ua.T,Va.T)
        plt.gca().set_aspect(1.)
        plt.colorbar()
        plt.title(f'Velocity at t={k*dt:.2f}')
        writer.grab_frame()
    
    # update progress bar
    if np.floor(51*k/Nit)>np.floor(51*(k-1)/Nit): 
        print('.',end='')

# finalise progress bar
print(' done. Iterations k=%i time=%.2f' % (k,k*dt))
toc()

if (ig>0):
    writer.finish()


#%% Visualization of the flow fiels at the end time

Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
Va = np.vstack((vW,avg(np.hstack((vS,V,
                                  vN)),0),vE));
plt.figure()
plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20)
plt.quiver(x,y,Ua.T,Va.T)
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title(f'Velocity at t={k*dt:.2f}')
plt.savefig('velocity.png')
plt.show()

#%% Compute divergence on cell centres

# compute divergence on cell centres
div = (np.diff( np.vstack( (uW,U, uE)),axis=0)/hx + np.diff( np.hstack(( vS, V, vN)),axis=1)/hy)
plt.figure()
plt.pcolor(avg(x),avg(y),div.T,shading='nearest')
plt.gca().set_aspect(1.)
plt.colorbar()
plt.title(f'Divergence at t={k*dt:.2f}')
plt.savefig('divergence.png')
plt.show()

#%% Analysis
"""
plt.figure()
plt.spy(Lp)
plt.show()

print(Lp.shape)
print(np.linalg.matrix_rank(Lp))
print(scl.null_space(Lp))
"""