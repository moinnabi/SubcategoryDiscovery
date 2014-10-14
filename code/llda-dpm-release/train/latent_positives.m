% get positive examples using latent detections
% we create virtual examples by flipping each image left to right
function [num_entries, num_examples, fusage, component_usage, scores, ...
          block_sums, block_counts] ...
  = latent_positives(t, iter, model, pos, fg_overlap, num_fp)

addpath(genpath('dpm-voc-release5/'));
conf = voc_config();
addpath(genpath('llda-dpm-release/'));


model.interval  = conf.training.interval_fg;
numpos          = length(pos);
pixels          = model.minsize * model.sbin / 2;
minsize         = prod(pixels);
fusage          = zeros(model.numfilters, 1);
component_usage = zeros(length(model.rules{model.start}), 1);
scores          = [];
block_sums      = [];
block_counts    = [];
num_entries     = 0;
num_examples    = 0;
batchsize       = max(1, 2*try_get_matlabpool_size());
% collect positive examples in parallel batches
for i = 1:batchsize:numpos
  % do batches of detections in parallel
  thisbatchsize = batchsize - max(0, (i+batchsize-1) - numpos);
  % data for batch
  clear('data');
  empties = cell(1, thisbatchsize);
  data = struct('boxdata', empties, 'pyra', empties);
  parfor k = 1:thisbatchsize
    j = i+k-1;
    msg = sprintf('%s %s: iter %d/%d: latent positive: %d/%d', ...
                  procid(), model.class, t, iter, j, numpos);
    % skip small examples
    if max(pos(j).sizes) < minsize
      data(k).boxdata = cell(length(pos(j).sizes), 1);
      fprintf('%s (all too small)\n', msg);
      continue;
    end

    % do whole image operations
    im = color(imreadx(pos(j)));
    [im, boxes] = croppos(im, pos(j).boxes);
    [pyra, model_dp] = gdetect_pos_prepare(im, model, boxes, fg_overlap);
    data(k).pyra = pyra;

    % process each box in the image
    num_boxes = size(boxes, 1);
    for b = 1:num_boxes
      % skip small examples
      if pos(j).sizes(b) < minsize
        data(k).boxdata{b} = [];
        fprintf('%s (%d: too small)\n', msg, b);
        continue;
      end
      fg_box = b;
      bg_boxes = 1:num_boxes;
      bg_boxes(b) = [];
      [ds, bs, trees] = gdetect_pos(data(k).pyra, model_dp, 1+num_fp, ...
                                    fg_box, fg_overlap, bg_boxes, 0.5);
      data(k).boxdata{b}.bs = bs;
      data(k).boxdata{b}.trees = trees;
      if ~isempty(bs)
        fprintf('%s (%d: comp %d  score %.3f)\n', msg, b, bs(1,end-1), bs(1,end));
      else
        fprintf('%s (%d: no overlap)\n', msg, b);
      end
    end
    model_dp = [];
  end
  % write feature vectors sequentially 
  for k = 1:thisbatchsize
    j = i+k-1;
    % write feature vectors for each box
    for b = 1:length(pos(j).dataids)
      if isempty(data(k).boxdata{b})
        continue;
      end
      dataid = pos(j).dataids(b);
      [bs, block_sums, block_counts] ...
          = gdetect_write(data(k).pyra, model, data(k).boxdata{b}.bs, ...
                          data(k).boxdata{b}.trees, true, dataid, ...
                          inf, inf, block_sums, block_counts);
      if ~isempty(bs)
        fusage = fusage + count_filter_usage(bs(1,:));
        component = bs(1,end-1);
        component_usage(component) = component_usage(component) + 1;
        num_entries = num_entries + size(bs, 1) + 1;
        num_examples = num_examples + 1;
        %loss = max([1; bs(:,end)]) - bs(1,end);
        %losses = [losses; loss];
        scores = [scores; bs(1,end)];
      end
    end
  end
end
