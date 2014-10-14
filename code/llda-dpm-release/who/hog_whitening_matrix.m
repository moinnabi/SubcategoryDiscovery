function [R, mu, S] = hog_whitening_matrix(bg, nx, ny, incr_lambda)
% function [R, mu] = hog_whitening_matrix(bg,nx,ny)
% Obtain whitening matrix and mean from a general HOG model   
% by a cholesky decompoition on a stationairy covariance matrix 
% feat' = (R')\(feat - mu) has zero mean and unit covariance
%
% bg.mu: HOG feature mean (nf by 1)
% bg.cov: covariance for k spatial offsets (nf by nf by k)
% bg.dxy: k spatial offsets (k by 2)
% bg.lambda: regularizer

if ~exist('incr_lambda', 'var')
  incr_lambda = true;
end

mu = repmat(bg.mu', ny*nx, 1);
mu = mu(:);
S = reconstruct_cov(bg, nx, ny);
my_eye = eye(size(S));
p = 1;
while p ~= 0
  % R'*R = S
  % R is diagonal + upper triangular
  l1 = 1 - bg.lambda;
  l2 = bg.lambda;
  [R, p] = chol(l1*S + l2*my_eye);
  if p ~= 0
    if incr_lambda
      warning('Increasing lambda');
      bg.lambda = bg.lambda + 0.01;
    else
      error('Would need to increase lambda');
    end
  end
end
