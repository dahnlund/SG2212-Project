function L = Lap(nx, ny, hx, hy)
    Dx = DD(nx, hx);
    Dy = DD(ny, hy);
    L = kron(speye(ny),Dx)+kron(Dy,speye(nx));
end