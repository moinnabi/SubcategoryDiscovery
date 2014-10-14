addpath(genpath('/home/moin/Desktop/UW/all_UW/cvpr_2015/code/dpm-voc-release5/'));
addpath(genpath('/home/moin/Desktop/UW/all_UW/cvpr_2015/code/bcp_release/'));
addpath(genpath('/home/moin/Desktop/Massimiliano/ihog-master/'));
addpath(genpath('phog/'));
addpath(genpath('llda-dpm-release/'));
run('vlfeat-0.9.19/toolbox/vl_setup');


run /home/moin/Desktop/UW/all_UW/cvpr_2015/code/bcp_release/setup.m
run /home/moin/Desktop/UW/all_UW/cvpr_2015/code/bcp_release/startup;

%%
voc_dir = '/home/moin/datasets/PASCALVOC/'; % CHANGE!!!
year = '2007'; set = 'train'; category = 'horse';
[pos, neg, impos] = pascal_data(category, year);
[~, voc_ng_train] = VOC_load(category,year,set,voc_dir);
load('data/horse_model/horse_final.mat','model');
%
I =[]; bb=[];
j = 1;
for i = 1 : 2 : size(pos(:),1)
    I{j} = pos(i).im;
    bb{j} = pos(i).boxes;
    j = j + 1;
end


%%
%Train Examplar-LDA for each patch (Query)
%addpath(genpath('bcp_release/'));
currentFolder = pwd;
VOCopts.localdir = [currentFolder,'/data/bcp_elda/'];
disp('orig_train_elda');
models = orig_train_elda(VOCopts, I, bb, 'ps', 'ng' , 0, 1);
for modl = 1:length(models)
    models_all{modl} = models{1,modl}.model;
end

%
[ng_detect] = run_patches_inside_wholeimage_on_negative(models_all,voc_ng_train);

%%
img_num = length(ng_detect);
numPatches = length(ng_detect{1}.ap_scores);
ap_score_all = zeros(img_num,numPatches);
for img = 1:img_num
    if  ~isempty(ng_detect{img})
        ap_score_all(img,:) = horzcat(ng_detect{img}.ap_scores{:});
    end
end
%Normalization over Patch ???
data_features = ap_score_all;
data_features=bsxfun(@minus,data_features,mean(data_features));
norm_temp=bsxfun(@rdivide,data_features,sqrt(sum(data_features.^2,2)));
ap_score_all_norm = norm_temp;

ng_score = zeros(img_num,numPatches);
for img = 1:img_num
    if  ~isempty(ng_detect{img})
         ng_score(img,:) = ap_score_all_norm(img,:);
    end
end





%%
blocksize = 8;
nx = 4; ny = 4;
scalefactor = sqrt(2);
hogNodes = load('feature_extraction/hogClusters.mat');

feature_pos = cell(size(pos(:),1),1);
for i = 1 : size(pos(:),1)
    i
    im = imread(pos(i).im); %im = color(im);
    bbox = pos(i).boxes;
    im_Crop = imcrop(im, [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)]);

    [feat, ~, ~, ~] = getHOGFeatures(im_Crop, blocksize, scalefactor, nx, ny);

    hogNodes = load('feature_extraction/hogClusters.mat');
    feature_pos{i} = getNearest(feat, hogNodes.centers);    
   
    
    %pyra = featpyramid(im_Crop, model);
%     roi = [bbox(2);bbox(4);bbox(1);bbox(3)]; %roi - Region Of Interest (ytop,ybottom,xleft,xright)
%     feature_pos(i,:) = anna_phog(im,bin,angle,L,roi);
end


feature_neg = zeros(size(neg(:),1).*size(pos(:),1),680);
for i = 1 : size(neg(:),1)
    im = voc_ng_train{i}.im; %im = color(im);
    for j = 1 : size(pos(:))
        bbox = ng_detect{i}.patches{j};
        %im_Crop = imcrop(im, [bbox(1) bbox(2) bbox(3)-bbox(1) bbox(4)-bbox(2)]);
        %pyra = featpyramid(im_Crop, model);
        roi = [bbox(2);bbox(4);bbox(1);bbox(3)]; %roi - Region Of Interest (ytop,ybottom,xleft,xright)
        feature_neg((i-1).*size(pos(:)+j,:)) = anna_phog(im,bin,angle,L,roi);
        %toc
    end
end