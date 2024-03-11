function A = DD(n,h)
% DD(n,h)
%
% One-dimensional finite-difference derivative matrix 
% of size n times n for second derivative:
%
% This function belongs to SG2212.m

    A = -eye(n) + [zeros(n-1,1),eye(n-1);zeros(1,n)];
    A = (A+A')/h^2;
    A(1, 1)=-1/h^2;
    A(n, n)=-1/h^2;
    A = sparse(A);
end
