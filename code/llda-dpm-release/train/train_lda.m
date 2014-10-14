function [model, obj_val_final] = ...
  train_lda(model, pos, warp, num_iters, fg_overlap, tag)

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2014 Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://www.cs.berkeley.edu/~rbg/latent)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

addpath(genpath('dpm-voc-release5/'));
conf = voc_config();
addpath(genpath('llda-dpm-release/'));


if ~exist('tag', 'var')
  tag = '';
end

bg = load(conf.features.path_to_bg);

% The feature vector cache will use a memory pool
% that can hold max_num feature vectors, each with
% a maximum byte size of single_byte_size*max_dim
[max_dim, max_nbls] = max_fv_dim(model);
bytelimit = conf.training.cache_byte_limit;
max_num = ceil(bytelimit / (conf.single_byte_size*max_dim));
fv_cache('init', max_num, max_dim, max_nbls);

[blocks, lb, rm, lm, cmps] = fv_model_args(model);
fv_cache('set_model', blocks, lb, rm, lm, cmps, 1, true);

obj_val = nan*ones(num_iters+1, 2);
for t = 1:num_iters
  fprintf('%s train lda iter: %d/%d\n', procid(), t, num_iters);
  
  % flush everything from cache
  fv_cache('shrink', int32([]));

  th = tic();
  % add new positives
  if warp
    [num_entries_added, num_examples_added, block_sums, block_counts] ...
        = warped_positives(t, model, pos);
    fusage = num_examples_added;
    component_usage = num_examples_added;
  else
    [num_entries_added, num_examples_added, fusage, component_usage, scores, ...
     block_sums, block_counts] ...
        = latent_positives(t, num_iters, model, pos, fg_overlap, 0);

    % save positive filter usage statistics
    model.stats.filter_usage = fusage;
    fprintf('\nFilter usage stats:\n');
    for i = 1:model.numfilters
      fprintf('  filter %d got %d/%d (%.2f%%) examples\n', ...
              i, fusage(i), num_examples_added, 100*fusage(i)/num_examples_added);
    end
    fprintf('\nComponent usage stats:\n');
    for i = 1:length(model.rules{model.start})
      fprintf('  component %d got %d/%d (%.2f%%) examples\n', ...
              i, component_usage(i), num_examples_added, ...
              100*component_usage(i)/num_examples_added);
    end
  end

  % Compute obj val prior to training
  blocks = fv_model_args(model);

  w = cat(1, blocks{:});
  pos_feat = cat(1, block_sums{:});
  obj_val(t+1,1) = w'*pos_feat / num_examples_added;
  if warp == 0
    model.thresh = min(scores);
    if abs(obj_val(t+1,1) - sum(scores)/num_examples_added) > 1e-5
      warning('obj val mismatch');
      fprintf('diff: %f\n', abs(obj_val(t+1,1) - sum(scores)/num_examples_added));
      %keyboard;
    end
  end

  % Train new filters  
  for b = 1:model.numblocks
    if model.blocks(b).type == block_types.Filter
      shape = model.blocks(b).shape;

      % LDA with unit norm in whitened space
      mu_pos                = reshape(block_sums{b}, shape);
      [w, bias, R, mu_bg]   = train_hog_lda_filter_w(bg.bg, mu_pos, block_counts(b));
      model.blocks(b).w     = w(:);
      model.blocks(b).bias  = bias;
      model.blocks(b).R     = R;
      model.blocks(b).mu_bg = mu_bg;
    end
  end

  model.debug = 0;
  model = normalize_components_w(model);

  % compute scores using HOG feature space filters
  if ~warp && model.debug
    scores = -inf(length(exs), 1);
    scores_white = -inf(length(exs), 1);
    for i = 1:length(exs)
      feat = cat(1, exs(i).blocks(:).f);
      bls  = cat(1, exs(i).blocks(:).bl);
      w    = cat(1, model.blocks(bls).w);
      w_w  = cat(1, model.blocks(bls).w_white);
      scores(i) = w'*feat;

      feat_w = {};
      for b = bls'
        if model.blocks(b).type == block_types.Filter
          shape = model.blocks(b).shape;
          f = exs(i).blocks(b).f(1:end-shape(1)*shape(2));
          feat_w{b} = (model.blocks(b).R')\(f - model.blocks(b).mu_bg);
        else
          feat_w{b} = exs(i).blocks(b).f;
        end
      end
      feat_w = cat(1, feat_w{:});
      scores_white(i) = w_w'*feat_w;
    end
    keyboard
  end
  th = toc(th);
  model.stats.pos_latent_time = [model.stats.pos_latent_time; th];

  % Compute obj val after training
  blocks = fv_model_args(model);
  w = cat(1, blocks{:});
  pos_feat = cat(1, block_sums{:});
  obj_val(t+1,2) = w'*pos_feat / num_examples_added;

  obj_ratio = obj_val(t+1,1)/obj_val(t,2);
  str1 = sprintf('Obj vals: before=%.4f  after=%.4f  ratio=%.4f', ...
                 obj_val(t,2), obj_val(t+1,1), obj_ratio);
  str2 = repmat('-', [1 length(str1)]);
  fprintf('%s\n%s\n%s\n\n', str2, str1, str2);
                
  if ~isempty(tag)
    % Save the intermediate model for debugging / inspection
    model_name = [model.class '_model_' tag '_' ...
                  num2str(t)];
    save([conf.paths.model_dir model_name], 'model');
  end

  if obj_ratio < 1.001
    fprintf('Convergence criteria met\n');
    break;
  end
end
obj_val_final = obj_val(end,end);

% collect filter usage statistics
function u = getfusage(bs)
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


% ------------------------------------------------------------------------
function [w, b, R, mu_bg] = train_hog_lda_filter_w(bg, mu_pos, tot_weight)
% ------------------------------------------------------------------------
% Computes a HOG LDA filter, but leaves it in the whitened space
% (for subsequent unit normalization in that space).

if ~isa(mu_pos, 'double')
  mu_pos = double(mu_pos);
end

% Remove truncation feature dim
mu_pos = mu_pos(:,:,1:end-1);

[ny nx nf] = size(mu_pos);

[R, mu_bg] = hog_whitening_matrix(bg, nx, ny, true);

% compute S^(-1/2)*(mu_pos-mu_bg) efficiently
w = R'\(mu_pos(:)-tot_weight*mu_bg);
b = 0;

% Add back truncation feature dimension
w = reshape(w, [ny nx nf]);
w(:,:,end+1) = 0;


% ------------------------------------------------------------------------
function model = normalize_components_w(model)
% ------------------------------------------------------------------------
% Unit normalize each component, in the whitened space, and then rotate
% back to unwhitened HOG filter space.

assert(model.type == model_types.MixStar);

[~, ~, ~, ~, comp_blocks] = fv_model_args(model);
for i = 1:length(comp_blocks)
  if isempty(comp_blocks{i}), continue, end;

  blocks_to_normalize = [];
  norm2               = 0;
  bias                = 0;
  for j = comp_blocks{i}(:)'
    bl = j+1;
    if model.blocks(bl).type == block_types.Filter
      norm2 = norm2 + model.blocks(bl).w'*model.blocks(bl).w;
      blocks_to_normalize = [blocks_to_normalize bl];
    end
    % for debugging
    model.blocks(bl).w_white = model.blocks(bl).w;
  end

  bias = 0;
  nrm = sqrt(norm2);
  for bl = blocks_to_normalize
    R = model.blocks(bl).R;
    mu_bg = model.blocks(bl).mu_bg;
    w = model.blocks(bl).w;
    shape = model.blocks(bl).shape;
    w = reshape(w, shape);
    w = w(:,:,1:end-1);
    model.blocks(bl).w_white = w(:) / nrm; % for debugging
    w = R\(w(:) / nrm);
    bias = bias + w'*mu_bg;
    w = reshape(w, [shape(1) shape(2) shape(3)-1]);
    w(:,:,end+1) = 0;
    model.blocks(bl).w = w(:);

    if ~model.debug
      model.blocks(bl).R = [];
      model.blocks(bl).mu_bg = [];
    end
  end
  obl = model.rules{model.start}(i).offset.blocklabel;
  model.blocks(obl).w = -bias / model.features.bias;
  % for debugging
  model.blocks(obl).w_white = 0;
end

% compute the max norm
max_nrm = -inf;
for i = 1:length(comp_blocks)
  if isempty(comp_blocks{i}), continue, end;
  norm2 = 0;
  for j = comp_blocks{i}(:)'
    bl = j+1;
    if model.blocks(bl).type == block_types.Filter
      norm2 = norm2 + model.blocks(bl).w'*model.blocks(bl).w;
    end
  end
  nrm = sqrt(norm2);
  if nrm > max_nrm
    max_nrm = nrm;
  end
end

% normalize so the max norm is 1
for i = 1:length(comp_blocks)
  if isempty(comp_blocks{i}), continue, end;

  blocks_to_normalize = [];
  for j = comp_blocks{i}(:)'
    bl = j+1;
    if model.blocks(bl).type == block_types.Filter
      blocks_to_normalize = [blocks_to_normalize bl];
    end
  end

  for bl = blocks_to_normalize
    model.blocks(bl).w = model.blocks(bl).w / max_nrm;
    % for debugging
    model.blocks(bl).w_white = model.blocks(bl).w_white / max_nrm;
  end
  obl = model.rules{model.start}(i).offset.blocklabel;
  model.blocks(obl).w = model.blocks(obl).w / max_nrm;
end
