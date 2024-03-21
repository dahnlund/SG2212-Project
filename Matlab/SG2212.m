% Navier-Stokes solver,
% adapted for course SG2212
% KTH Mechanics
%
% Depends on avg.m and DD.m
%
% Code version: 
% 20180222

clear all

J = jet(40^3);
J2 = zeros(40,3);
for i = 1:40
    j = floor(64000*((i-1)/40)^(1/3))+1;
    J2(i,:) = J(j,:);
end
clear J i j

%------------------------------------------
% dots = zeros(100, 100, 3); dots(:,1,1) = rand(100, 1); dots(:,1,2) = rand(100,1);
lid_driven_cavity=0;

if (lid_driven_cavity==1)
  % Parameters for test case I: Lid-driven cavity
  % The Richardson number is zero, i.e. passive scalar.
  
  Pr = 0.71;     % Prandtl number
  Re = 4000;      % Reynolds number
  Ri = 0.;       % Richardson number
  Ra = 1705;      % Rayleigh number
  
  dt = 0.0001;      % time step
  Tf = 50;       % final time
  Lx = 1;        % width of box
  Ly = 1;        % height of box
  Nx = 50;      % number of cells in x
  Ny = 50;      % number of cells in y
  ig = 100;      % number of iterations between output
  
  % Boundary and initial conditions:
  Utop = 1.; 
  Ubottom = 0.;
  % IF TEMPERATURE:
  Tbottom = 1.; Ttop = 0.;
  namp = 0.;
else
  % Parameters for test case II: Rayleigh-Bénard convection
  % The DNS will be stable for Ra=1705, and unstable for Ra=1715 
  % (Theoretical limit for pure sinusoidal waves 
  % with L=2.01h: Ra=1708)
  % Note the alternative scaling for convection problems.
    
  Pr = 0.71;     % Prandtl number
  Ra = 20000;      % Rayleigh number

  Re = 1./Pr;    % Reynolds number
  Ri = Ra*Pr;    % Richardson number
  
  dt = 0.00005;   % time step
  Tf = 20;       % final time
  Lx = 10.;      % width of box
  Ly = 1;        % height of box
  Nx = 120;      % number of cells in x
  Ny = 20;      % number of cells in y
  ig = 100;      % number of iterations between output
  
  % Boundary and initial conditions:
  Utop = 0; Ttop = 1.;
  Ubottom = 0.; Tbottom = 0;
  % IF TEMPERATURE:
  Tbottom = 1.; Ttop = 0.;
  namp = 0.;
end


%-----------------------------------------

% Number of iterations
Nit = Tf/dt;
% Spatial grid: Location of corners 
x = linspace(0, Lx, Nx+1); 
y = linspace(0, Ly, Ny+1); 
% Grid spacing

dx = x(2)-x(1);
dy = y(2)-y(1);
% Boundary conditions: 
uN = x*0+Utop; uN = uN';    vN = avg(x, 2)*0; vN = vN';
uW = avg(y, 2)*0;           vW = y*0;
uS = x*0+Ubottom; uS = uS'; vS = avg(x, 2)*0; vS = vS';
uE = avg(y, 2)*0;           vE = y*0;
tN = avg(x,2)*0+Ttop;tN=tN'; tS = avg(x,2)*0+Tbottom; tS = tS';
% Initial conditions
U = zeros(Nx-1, Ny); V = zeros(Nx, Ny-1);
% linear profile for T with random noise
T = (Ttop-Tbottom)*ones(size(avg(x',1)))*avg(y,2) + Tbottom + namp*rand(Nx,Ny);
% Time series
tser = [];
Tser = [];

%-----------------------------------------

% Compute system matrices for pressure 
% First set homogeneous Neumann condition all around
% Laplace operator on cell centres: Fxx + Fyy
Lp = Lap(Nx, Ny, dx, dy);
% Set one Dirichlet value to fix pressure in that point
Lp(1,:) = zeros(1, size(Lp, 2)) ; Lp(1,1) = 1 ;
Mp = Lp^-1;
% Here you can pre-compute the LU decomposition
% [LLp,ULp] = lu(Lp);
%-----------------------------------------

% Progress bar (do not replace the ... )
fprintf(...
    '[         |         |         |         |         ]\n')

%-----------------------------------------

% Main loop over iterations
f1 = figure("Position",[30 30 1000 500]);
Vid = VideoWriter(strcat("result_simu_PV.mp4"),'MPEG-4'); open(Vid)
%Vid2 = VideoWriter(strcat("result_simu_V.mp4"),'MPEG-4'); open(Vid2)
for k = 1:Nit
     
   % include all boundary points for u and v (linear extrapolation
   % for ghost cells) into extended array (Ue,Ve)
   Ue = [uW ; U ; uE];
   Ve = [vS, V, vN];
   Ue = [2*uS-Ue(:,1), Ue, 2*uN-Ue(:,end)];
   Ve = [2*vW-Ve(1,:); Ve; 2*vE-Ve(end,:)];
   
   % averaged (Ua,Va) of u and v on corners
   Ua = avg(Ue, 2);
   Va = avg(Ve, 1);
   
   % construct individual parts of nonlinear terms
   dUVdx = 1/dx*diff(Ua.*Va,1,1);
   dUVdy = 1/dy*diff(Ua.*Va, 1, 2);
   Ub = avg(Ue(:,2:end-1),1);
   Vb = avg(Ve(2:end-1,:), 2);
   dU2dx = 1/dx*diff(Ub.^2, 1, 1);
   dV2dy = 1/dy*diff(Vb.^2, 1, 2);
   % treat viscosity explicitly
   viscu = diff(Ue(:,2:end-1),2,1 )/dx^2 + diff( Ue(2:end-1,:),2,2 )/dy^2;
   viscv = diff(Ve(:,2:end-1),2,1 )/dx^2 + diff( Ve(2:end-1,:),2,2 )/dy^2;
   
   % buoyancy term
   % IF TEMPERATURE: fy = ...
         
   % compose final nonlinear term + explicit viscous terms
   U = U + dt*( viscu/Re -(dU2dx + dUVdy(2:end-1,:)) );
   V = V + dt*( viscv/Re -(dV2dy + dUVdx(:,2:end-1)) );
   
   % pressure correction, Dirichlet P=0 at (1,1)
   rhs = ((diff( [uW ; U ; uE], 1, 1)/dx) + diff( [vS, V, vN], 1, 2)/dy)/dt;
   rhs(1, 1) = 0;
   rhs = reshape(rhs,Nx*Ny,1);
   P = Mp*rhs;
   % alternatively, you can use the pre-computed LU decompositon
   % P = ...;
   % or gmres
   % P = gmres(Lp, rhs, [], tol, maxit);
   % or as another alternative you can use GS / SOR from homework 6
	% [PP, r] = GS_SOR(omega, Nx, Ny, hx, hy, L, f, p0, tol, maxit);
   P = reshape(P,Nx,Ny);
   
   % apply pressure correction
   U = U - dt*diff(P, 1, 1)/dx;
   V = V - dt*diff(P, 1, 2)/dy;
   
   % Temperature equation
   % IF TEMPERATURE:
   Te = [2*tS-T(:,1), T, 2*tN-T(:,end)];
   Te = [Te(2,:); Te; Te(end-1, :)];
   % IF TEMPERATURE:
   Tu = avg(avg(Te, 1), 2).*avg(Ue, 2);
   % IF TEMPERATURE:
   Tv = avg(avg(Te, 1), 2).*avg(Ve, 1);
   % IF TEMPERATURE: 
   H = -avg(diff(Tu, 1, 1), 2)/dx-avg(diff(Tv, 1, 2), 1)/dy+(diff(Te(:, 2:end-1), 2, 1)/dx^2 + diff(Te(2:end-1, :), 2, 2)/dy^2);
   % IF TEMPERATURE:
   T = T + dt*H;

   V = V+Ra*Pr*avg(T, 2)*dt;
   
   %-----------------------------------------
   
   % progress bar
   if floor(51*k/Nit)>floor(51*(k-1)/Nit), fprintf('.'), end
   
   % plot solution if needed
   if k==1||floor(k/ig)==k/ig

     % compute divergence on cell centres
     % if (1==1)
     %   div = diff([uW;U;uE], 1, 1)/dx + diff([vS, V, vN],1,2)/dy;
     % 
     %   figure(1);clf; hold on;
     %   contourf(avg(x,2),avg(y,2),div');colorbar
     %   axis equal; axis([0 Lx 0 Ly]);
     %   title(sprintf('divergence at t=%g',k*dt))
     %   drawnow
     % end 
     
     % compute velocity on cell corners
     Ua = avg(Ue, 2);
     Va = avg(Ve, 1);
     Len = sqrt(Ua.^2+Va.^2+eps);
     
     % figure(); clf; hold on
     % ind = min(k, 100);
     % for i = 1:100
     % 
     % end

     figure(f1); clf;
     if lid_driven_cavity == 1
        ax = subplot(1, 3, 2);
     else
        ax = subplot(3, 1, 2);
     end
     hold on;
     contourf(ax, avg(x,2),avg(y,2),P', 50, 'LineColor','none');colorbar
     %quiver(ax, x,y,(Ua./Len)',(Va./Len)',.4,'k-')
     ax.CLim = [-5 5]; ax.Colormap = bone(50);
     axis equal

     if lid_driven_cavity == 1
        ax = subplot(1, 3, 1);
     else
        ax = subplot(3, 1, 1);
     end
     title(sprintf('Solution at t=%g',k*dt))
     hold on;
     contourf(ax, x,y,sqrt(Ua.^2+Va.^2)',20,'LineColor','none');colorbar
     q = quiver(ax, x,y,(Ua./Len)',(Va./Len)',.4,'k-');
     axis equal; axis([0 Lx 0 Ly]); ax.Colormap = jet();

     if lid_driven_cavity == 1
        ax = subplot(1, 3, 3);
     else
        ax = subplot(3, 1, 3);
     end
     hold on;
     contourf(ax, avg(x,2),avg(y,2),T', 50, 'LineColor','none');colorbar
     %quiver(ax, x,y,(Ua./Len)',(Va./Len)',.4,'k-')
     ax.CLim = [min(Tbottom, Ttop), max(Tbottom, Ttop)]; ax.Colormap = flip(hot(50));
     axis equal

     drawnow
     frame = getframe(gcf);
     writeVideo(Vid, frame);
     
     % IF TEMPERATURE: % compute temperature on cell corners
     % IF TEMPERATURE: Ta = ...
     
     % IF TEMPERATURE: figure(3); clf; hold on;
     % IF TEMPERATURE: contourf(x,y,Ta',20,'k-');colorbar
     % IF TEMPERATURE: quiver(x,y,(Ua./Len)',(Va./Len)',.4,'k-')
     % IF TEMPERATURE: axis equal; axis([0 Lx 0 Ly]);
     % IF TEMPERATURE: title(sprintf('T at t=%g',k*dt))
     % IF TEMPERATURE: drawnow
     
     % Time history
     % if (1==1)
     %   figure(4); hold on;
     %   tser = [tser k*dt];
     %   Tser = [Tser Ue(ceil((Nx+1)/2),ceil((Ny+1)/2))];
     %   plot(tser,abs(Tser))
     %   title(sprintf('Probe signal at x=%g, y=%g',...
     %         x(ceil((Nx+1)/2)),y(ceil((Ny+1)/2))))
     %   set(gca,'yscale','log')
     %   xlabel('time t');ylabel('u(t)')
     % end
   end
end
close(Vid)
%close(Vid2)
fprintf('\n')
