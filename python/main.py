#%%
'''Load dependencies'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.animation
import math 
import scipy.sparse as sp
import scipy.linalg as scl
from scipy.io import savemat, loadmat
from tqdm import tqdm
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
from src import tic, toc, extract

#%% Simulation parameters

lid_driven_cavity = True
anim = False  # Save animation
compare_with_openfoam = False
save_matrices = False
use_stored_data = False


if lid_driven_cavity:
    Pr = 0.71     #Prandtl number
    Ra = 1705      # Rayleigh number
    Ri = 0
    # Specify which Re cases to run:
    cases = [25]  #Re-number
    # Ri = 0. 
    dt = 0.001
    Tf = 50
    Lx = 1.
    Ly = 1.
    Nx = 30
    Ny = 30
    namp = 0.
    ig = 20
else:
    Pr = 0.71     #Prandtl number
    Ra = 20000      # Rayleigh number
    cases = [1/Pr]    # Reynolds number
    Ri = Ra*Pr    # Richardson number
    dt = 0.0005   # time step
    Tf = 2       # final time
    Lx = 10      # width of box
    Ly = 1        # height of box
    Nx = 120      # number of cells in x
    Ny = 20      #number of cells in y
    namp = 0.
    ig = 100      # number of iterations between output

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
Utop = 1; Ttop = 0.; Tbottom = 0.;
uN = x*0 + Utop;  uN = uN[:,np.newaxis];    vN = avg(x)*0;    vN = vN[:,np.newaxis];
uS = x*0;  uS = uS[:,np.newaxis];         vS = avg(x)*0;  vS = vS[:,np.newaxis];
uW = avg(y)*0;  uW = uW[np.newaxis,:];       vW = y*0;  vW = vW[np.newaxis,:];
uE = avg(y)*0;  uE = uE[np.newaxis,:];       vE = y*0;    vE = vE[np.newaxis,:];



#%% Pressure correction and pressure Poisson equation

# Compute system matrices for pressure 
# Laplace operator on cell centres: Fxx + Fyy
# First set homogeneous Neumann condition all around
Lp = sp.kron(sp.eye(Ny, format = 'csc'), DD(Nx,hx), format = 'csc') \
    + sp.kron(DD(Ny,hy), sp.eye(Nx, format = 'csc'), format = 'csc')
# Set one Dirichlet value to fix pressure in that point
Lp[0,:] =0; Lp[0,0] = 1;
Lps_lu = splu(Lp)


#%% Main time-integration loop. Write output file "cavity.mp" if

if (ig>0) and anim:
    metadata = dict(title='Lid-driven cavity', artist='SG2212')
    writer = matplotlib.animation.FFMpegWriter(fps=15, metadata=metadata)
    matplotlib.use("Agg")
    fig=plt.figure()
    writer.setup(fig,"cavity.mp4",dpi=200)

tic()

#Define probe
uvel = np.zeros((Nit+1, len(cases)))

for i, Re in enumerate(cases):

    # Initial conditions

    U = np.zeros((Nx-1,Ny))
    V = np.zeros((Nx,Ny-1))

    T = ((Tbottom - Ttop) * np.ones(avg(x).shape)[np.newaxis,:] * avg(y)[:, np.newaxis]).T \
        + Tbottom + namp*np.random.rand(Nx,Ny)

    print(f"Running case for Re = {Re}")
    for k in tqdm(range(Nit), desc="Iterations"):
        if use_stored_data:
            print("Using stored data -> exiting loop")
            break
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
        P = np.reshape(P.T, (Ny,Nx)).T

        # apply pressure correction
        U = U - dt*np.diff(P, axis = 0)/hx;
        V = V - dt*np.diff(P, axis = 1)/hy; 

        # Temperature equation
        # IF TEMPERATURE:
        Te = np.hstack(((2*Tbottom-T[:,0])[:,np.newaxis], T, (2*Ttop-T[:,-1])[:,np.newaxis]))
        Te = np.vstack((Te[1,:], Te, Te[-2, :]))

        Tu = avg(avg(Te, 0), 1)*avg(Ue, 1)

        Tv = avg(avg(Te, 0), 1)*avg(Ve, 0)

        H = -avg(np.diff(Tu, axis = 0), 1)/hx-avg(np.diff(Tv, axis = 1), 0)/hy\
            +(np.diff(Te[:, 1:-1], axis = 0, n = 2)/hx**2 + np.diff(Te[1:-1, :], axis = 1, n = 2)/hy**2)
        T = T + dt*H

        V = V+Ra*Pr*avg(T, 1)*dt;
        
        if (ig>0 and np.floor(k/ig)==k/ig and anim):
            Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
            Va = np.vstack((vW,avg(np.hstack((vS,V,
                                            vN)),0),vE));
            plt.clf()
            normalizer = matplotlib.colors.Normalize(0,0.7)
            plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20, norm = normalizer, cmap = "inferno")
            plt.quiver(x,y,Ua.T,Va.T)
            plt.gca().set_aspect(1.)
            plt.colorbar(norm = normalizer, cmap = "inferno")
            plt.title(f'Velocity at t={k*dt:.2f}, Re = {Re}, N = {Nx}')
            writer.grab_frame()
        
        uvel[k+1, i] = U[int(Nx/2), int(Ny/2)] #Save data in the middle of the domain
        

    # finalise progress bar
    print(' done. Iterations k=%i time=%.2f' % (k+1,k*dt))
    toc()

    if (ig>0) and anim:
        writer.finish()
    #%% Visualization of the flow fiels at the end time

    if use_stored_data:
        mat = loadmat(f'U_V_RE{Re}.mat')
        # Access variables from the .mat file
        Ua= mat['Ua']
        Va = mat['Va']
        x = np.linspace(0,Lx, Ua.shape[0])
        y = np.linspace(0,Ly, Va.shape[1])

    else:
        Ua = np.hstack( (uS,avg(np.vstack((uW,U,uE)),1),uN));
        Va = np.vstack((vW,avg(np.hstack((vS,V,
                                            vN)),0),vE));
        

        T = np.hstack((T, (2*Ttop-T[:,-1])[:,np.newaxis]))
        T = np.vstack((T, T[-2, :]))

    plt.figure()
    normalizer = matplotlib.colors.Normalize(0,0.7)
    plt.contourf(x,y,np.sqrt(Ua**2+Va**2).T,20,norm = normalizer, cmap = "inferno")
    plt.quiver(x,y,Ua.T,Va.T,norm = normalizer, cmap = "inferno")
    #plt.scatter(x[int(Nx/2)], y[int(Ny/2)], color='red') 
    plt.gca().set_aspect(1.)
    plt.colorbar(norm = normalizer, cmap = "inferno")
    plt.title(f'Velocity at t={k*dt:.2f}, Re = {Re}, N = {Nx}')
    plt.savefig(f'./plots/velocity_RE{Re}.png')

    
    plt.figure()
    normalizer = matplotlib.colors.Normalize(0,0.7)
    plt.contourf(x,y,T.T,20,norm = normalizer, cmap = "inferno")
    plt.quiver(x,y,Ua.T,Va.T,norm = normalizer, cmap = "inferno")
    #plt.scatter(x[int(Nx/2)], y[int(Ny/2)], color='red') 
    plt.gca().set_aspect(1.)
    plt.colorbar(norm = normalizer, cmap = "inferno")
    plt.savefig(f'./plots/temp.png')
    
    # Save Ua and Va to a .mat file
    if save_matrices:
        data = {"Ua": Ua, "Va": Va}
        savemat(f"U_V_RE{Re}.mat", data)

    '''Compare with openfoam solution'''
    if compare_with_openfoam:
        list1 = ['A', 'B', 'C']
        Uo, Vo = extract(list1[i])
        vel_amp = np.sqrt(Ua**2+Va**2).T[:-1, :-1]
        vel_amp_o = np.sqrt(Uo**2+Vo**2)[:-1, :-1]
        #Print relative norm
        rel_error = np.linalg.norm(vel_amp-vel_amp_o)/np.linalg.norm(vel_amp)
        print(f"Relative error: {rel_error}")

        plt.figure()
        normalizer = matplotlib.colors.Normalize(0,0.7)
        plt.contourf(x,y,np.sqrt(Uo**2+Vo**2),20,norm = normalizer, cmap = "inferno")
        plt.quiver(x,y,Ua.T,Va.T,norm = normalizer, cmap = "inferno")
        plt.gca().set_aspect(1.)
        plt.colorbar(norm = normalizer, cmap = "inferno")
        plt.title(f'Velocity at t={k*dt:.2f}, Re = {Re}, N = {Nx}')
        plt.savefig(f'./plots/velocity_RE{Re}_OF.png')

        # Over_line plot
        line = np.diagonal(vel_amp)
        line_OF = np.diagonal(vel_amp_o)

        plt.figure()
        plt.title(f"Plot over diagonal line, Re = {Re}")
        plt.plot(line, label = "Python solution")
        plt.plot(line_OF, label = "OpenFOAM solution")
        plt.xlabel("diagonal index")
        plt.ylabel("Velocity magnitude")
        plt.legend()
        plt.grid()
        plt.savefig(f'./plots/overline_RE{Re}.png')
        


plt.figure()
plt.plot(np.linspace(0,Tf,int(Tf/dt)+1), uvel)
plt.ylabel("U")
plt.xlabel("time")
plt.legend(cases)
plt.grid()
plt.savefig("./plots/plot1.png")
