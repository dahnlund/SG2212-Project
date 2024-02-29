% Navier-Stokes solver,
% adapted for course SG2212
% KTH Mechanics
%
% Depends on avg.m and DD.m
%
% Code version: 
% 20180222

clear all

%------------------------------------------

lid_driven_cavity=1;

if (lid_driven_cavity==1)
  % Parameters for test case I: Lid-driven cavity
  % The Richardson number is zero, i.e. passive scalar.
  
  Pr = 0.71;     % Prandtl number
  Re = ...;      % Reynolds number
  Ri = 0.;       % Richardson number
  
  dt = ...;      % time step
  Tf = 20;       % final time
  Lx = 1;        % width of box
  Ly = 1;        % height of box
  Nx = ...;      % number of cells in x
  Ny = ...;      % number of cells in y
  ig = 200;      % number of iterations between output
  
  % Boundary and initial conditions:
  Utop = 1.; 
  Ubottom = 0.;
  % IF TEMPERATURE: Tbottom = 1.; Ttop = 0.;
  % IF TEMPERATURE: namp = 0.;
else
  % Parameters for test case II: Rayleigh-Bénard convection
  % The DNS will be stable for Ra=1705, and unstable for Ra=1715 
  % (Theoretical limit for pure sinusoidal waves 
  % with L=2.01h: Ra=1708)
  % Note the alternative scaling for convection problems.
    
  Pr = 0.71;     % Prandtl number
  Ra = ...;      % Rayleigh number

  Re = 1./Pr;    % Reynolds number
  Ri = Ra*Pr;    % Richardson number
  
  dt = ...;   % time step
  Tf = 20;       % final time
  Lx = 10.;      % width of box
  Ly = 1;        % height of box
  Nx = ...;      % number of cells in x
  Ny = ...;      % number of cells in y
  ig = 200;      % number of iterations between output
  
  % Boundary and initial conditions:
  Utop = 0.; 
  Ubottom = 0.;
  % IF TEMPERATURE: Tbottom = 1.; Ttop = 0.;
  namp = 0.1;
end


%-----------------------------------------

% Number of iterations
Nit = ...
% Spatial grid: Location of corners 
x = linspace( ... ); 
y = linspace( ... ); 
% Grid spacing

dx = ...
dy = ...
% Boundary conditions: 
uN = x*0+Utop;    vN = avg(x,2)*0;
uS = ...          vS = ...
uW = ...          vW = ...
uE = ...          vE = ...
tN = ...          tS = ...
% Initial conditions
U = ...; V = ...;
% linear profile for T with random noise
% IF TEMPERATURE: T = ... + namp*rand(Nx,Ny)
% Time series
tser = [];
Tser = [];

%-----------------------------------------

% Compute system matrices for pressure 
% First set homogeneous Neumann condition all around
% Laplace operator on cell centres: Fxx + Fyy
Lp = kron(speye(Ny), ... ) + kron( ... ,speye(Nx));
% Set one Dirichlet value to fix pressure in that point
Lp(1,:) = ... ; Lp(1,1) = ... ;
% Here you can pre-compute the LU decomposition
% [LLp,ULp] = lu(Lp);
%-----------------------------------------

% Progress bar (do not replace the ... )
fprintf(...
    '[         |         |         |         |         ]\n')

%-----------------------------------------

% Main loop over iterations

for k = 1:Nit
     
   % include all boundary points for u and v (linear extrapolation
   % for ghost cells) into extended array (Ue,Ve)
   Ue = ...
   Ve = ...
   
   % averaged (Ua,Va) of u and v on corners
   Ua = ...
   Va = ...
   
   % construct individual parts of nonlinear terms
   dUVdx = ...
   dUVdy = ...
   dU2dx = ...
   dV2dy = ...
   
   % treat viscosity explicitly
   viscu = ...
   viscv = ...
   
   % buoyancy term
   % IF TEMPERATURE: fy = ...
         
   % compose final nonlinear term + explicit viscous terms
   U = U + dt/Re*... - dt*(...);
   V = V + dt/Re*... - dt*(...) + % IF TEMPERATURE: dt*fy;
   
   % pressure correction, Dirichlet P=0 at (1,1)
   rhs = (diff( ... )/dx + diff( ... )/dy)/dt;
   rhs = reshape(rhs,Nx*Ny,1);
   rhs(1) = ...
   P = Lp\rhs;
   % alternatively, you can use the pre-computed LU decompositon
   % P = ...;
   % or gmres
   % P = gmres(Lp, rhs, [], tol, maxit);
   % or as another alternative you can use GS / SOR from homework 6
	% [PP, r] = GS_SOR(omega, Nx, Ny, hx, hy, L, f, p0, tol, maxit);
   P = reshape(P,Nx,Ny);
   
   % apply pressure correction
   U = U - ...
   V = V - ...
   
   % Temperature equation
   % IF TEMPERATURE: Te = ... 
   % IF TEMPERATURE: Tu = ...
   % IF TEMPERATURE: Tv = ...
   % IF TEMPERATURE: H = ...
   % IF TEMPERATURE: T = T + dt*H;
   
   %-----------------------------------------
   
   % progress bar
   if floor(51*k/Nit)>floor(51*(k-1)/Nit), fprintf('.'), end
   
   % plot solution if needed
   if k==1|floor(k/ig)==k/ig

     % compute divergence on cell centres
     if (1==1)
       div = diff([uW;U;uE])/dx + diff([vS' V vN'],1,2)/dy;

       figure(1);clf; hold on;
       contourf(avg(x,2),avg(y,2),div');colorbar
       axis equal; axis([0 Lx 0 Ly]);
       title(sprintf('divergence at t=%g',k*dt))
       drawnow
     end 
     
     % compute velocity on cell corners
     Ua = ...
     Va = ...
     Len = sqrt(Ua.^2+Va.^2+eps);

     figure(2);clf;hold on;
     %contourf(avg(x,2),avg(y,2),P');colorbar
     contourf(x,y,sqrt(Ua.^2+Va.^2)',20,'k-');colorbar
     quiver(x,y,(Ua./Len)',(Va./Len)',.4,'k-')
     axis equal; axis([0 Lx 0 Ly]);
     title(sprintf('u at t=%g',k*dt))
     drawnow
     
     % IF TEMPERATURE: % compute temperature on cell corners
     % IF TEMPERATURE: Ta = ...
     
     % IF TEMPERATURE: figure(3); clf; hold on;
     % IF TEMPERATURE: contourf(x,y,Ta',20,'k-');colorbar
     % IF TEMPERATURE: quiver(x,y,(Ua./Len)',(Va./Len)',.4,'k-')
     % IF TEMPERATURE: axis equal; axis([0 Lx 0 Ly]);
     % IF TEMPERATURE: title(sprintf('T at t=%g',k*dt))
     % IF TEMPERATURE: drawnow
     
     % Time history
     if (1==1)
       figure(4); hold on;
       tser = [tser k*dt];
       Tser = [Tser Ue(ceil((Nx+1)/2),ceil((Ny+1)/2))];
       plot(tser,abs(Tser))
       title(sprintf('Probe signal at x=%g, y=%g',...
             x(ceil((Nx+1)/2)),y(ceil((Ny+1)/2))))
       set(gca,'yscale','log')
       xlabel('time t');ylabel('u(t)')
     end
   end
end
fprintf('\n')
