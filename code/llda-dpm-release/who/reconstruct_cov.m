function S = reconstruct_cov(bg, nx, ny)
% S = reconstruct_cov(nx, ny, bg_cov, dxy)
% S = n x n 
% n = ny * nx * nf

dxy    = bg.dxy;
bg_cov = bg.cov;
k      = size(dxy, 1);
nf     = size(bg_cov, 1);
n      = ny*nx;  
S      = zeros(nf, nf, n, n);

for x1 = 1:nx
  for y1 = 1:ny
    i1 = (x1-1)*ny + y1;
    for i = 1:k
      x = dxy(i,1);
      y = dxy(i,2);
      x2 = x1 + x;        
      y2 = y1 + y;
      if x2 >= 1 && x2 <= nx && y2 >= 1 && y2 <= ny
        i2 = (x2-1)*ny + y2;
        S(:,:,i1,i2) = bg_cov(:,:,i); 
      end
      x2 = x1 - x;        
      y2 = y1 - y;
      if x2 >= 1 && x2 <= nx && y2 >= 1 && y2 <= ny
        i2 = (x2-1)*ny + y2; 
        S(:,:,i1,i2) = bg_cov(:,:,i)'; 
      end
    end
  end
end

% Permute [nf nf n n] to [n nf n nf]
S = permute(S, [3 1 4 2]);
S = reshape(S, [n*nf n*nf]);

% Make sure returned matrix is close to symmetric
%assert(sum(sum(abs(S - S'))) < 1e-5);

S = (S+S')/2;
