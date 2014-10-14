function u = count_filter_usage(bs)
% collect filter usage statistics
numfilters = floor(size(bs, 2)/4);
u = zeros(numfilters, 1);
nbs = size(bs,1);
for i = 1:numfilters
  x1 = bs(:,1+(i-1)*4);
  y1 = bs(:,2+(i-1)*4);
  x2 = bs(:,3+(i-1)*4);
  y2 = bs(:,4+(i-1)*4);
  ndel = sum((x1 == 0) .* (x2 == 0) .* (y1 == 0) .* (y2 == 0));
  u(i) = nbs - ndel;
end
