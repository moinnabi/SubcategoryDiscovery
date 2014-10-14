function [ng_detect] = run_patches_inside_wholeimage_on_negative(models_all,voc_ng_train)
% dataset should be stored in a cell
ng_detect = cell(1,length(voc_ng_train));

disp('doing part inference');
parfor i = 1:length(voc_ng_train)
    disp([int2str(i),'/',int2str(length(voc_ng_train))])
    im_current = imread(voc_ng_train(i).im);

    [im_h, im_w, ~] = size(im_current);
%     ro = 0.07; %inspired by Santosh cvpr'14
%     bbox_current = [ro*im_w ro*im_h (1-ro)*im_w (1-ro)*im_h];
    bbox_current = [1 1 im_w im_h];
    [ng_detect{i}.ap_scores, ng_detect{i}.sp_scores, ng_detect{i}.patches] = part_inference(im_current, models_all, bbox_current);
end