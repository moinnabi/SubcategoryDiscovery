% get positive examples by warping positive bounding boxes
% we create virtual examples by flipping each image left to right
function [num_entries, num_examples, block_sums, block_counts] ...
  = warped_positives(t, model, pos)
% assumption: the model only has a single structure rule 
% of the form Q -> F.
numpos = length(pos);
warped = warppos(model, pos);
fi = model.symbols(model.rules{model.start}.rhs).filter;
fbl = model.filters(fi).blocklabel;
obl = model.rules{model.start}.offset.blocklabel;
pixels = model.filters(fi).size * model.sbin / 2;
minsize = prod(pixels);
num_entries = 0;
num_examples = 0;
is_belief = 1;
is_mined = 0;
loss = 0;

block_sums = cell(model.numblocks, 1);
block_counts = zeros(model.numblocks, 1);
for i = 1:model.numblocks
  block_sums{i} = zeros(model.blocks(i).dim, 1);
end

for i = 1:numpos
  fprintf('%s %s: iter %d: warped positive: %d/%d\n', ...
          procid(), model.class, t, i, numpos);
  bbox = [pos(i).x1 pos(i).y1 pos(i).x2 pos(i).y2];
  % skip small examples
  if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
    continue;
  end    
  % get example
  im = warped{i};
  feat = features(double(im), model.sbin);
  key = [i 0 0 0];
  bls = [obl; fbl] - 1;
  feat = [model.features.bias; feat(:)];
  fv_cache('add', int32(key), int32(bls), single(feat), ...
                  int32(is_belief), int32(is_mined), loss); 
  write_zero_fv(true, key);
  num_entries = num_entries + 2;
  num_examples = num_examples + 1;

  block_sums{obl} = block_sums{obl} + feat(1);
  block_sums{fbl} = block_sums{fbl} + feat(2:end);
  block_counts(obl) = block_counts(obl) + 1;
  block_counts(fbl) = block_counts(fbl) + 1;
end
