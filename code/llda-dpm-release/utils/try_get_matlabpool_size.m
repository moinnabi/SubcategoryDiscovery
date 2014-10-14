function s = try_get_matlabpool_size()
try
  s = matlabpool('size');
catch
  s = 0;
end
