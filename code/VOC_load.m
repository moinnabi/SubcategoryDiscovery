function [voc_ps, voc_ng] = VOC_load(cls,year,set,voc_dir)

%Initial parameter
VOC_root = [voc_dir,'VOC',year,'/VOCdevkit'];
%addpath([VOC_root '/VOCcode']);
run([VOC_root '/VOCinit.m']);
%VOCinit; %set for 2007
%year = '2007';
% i =13; %horse
% cls=VOCopts.classes{i};

%Load Positive
[ids, gt] = textread(sprintf(VOCopts.clsimgsetpath, cls, set), '%s %d');

ids = ids(gt == 1);
  pos    = [];
  numpos = 0;
  dataid   = 0;
  for i = 1:length(ids);
    tic_toc_print('%s: parsing positive (%s %s): %d/%d\n', ...
                  cls, set, '2007', i, length(ids));
      dataid             = dataid + 1;
      numpos             = numpos+1;
      pos(numpos).im     = sprintf(VOCopts.imgpath, ids{i});
      pos(numpos).flip   = false;
      pos(numpos).dataid = dataid;    
  end
voc_ps = pos;

%Load Negative
[ids, gt] = textread(sprintf(VOCopts.clsimgsetpath, cls, set), '%s %d');
ids = ids(gt == -1);
  neg    = [];
  numpos = 0;
  dataid   = 0;
  for i = 1:length(ids);
    tic_toc_print('%s: parsing Negative (%s %s): %d/%d\n', ...
                  cls, set, '2007', i, length(ids));
      dataid             = dataid + 1;
      numpos             = numpos+1;
      neg(numpos).im     = sprintf(VOCopts.imgpath, ids{i});
      neg(numpos).flip   = false;
      neg(numpos).dataid = dataid;    
  end
%
%for random selecting of negative set in the same size as positive set
% rend_ind = randi(length(neg),[1,length(ps)]);
% ng = neg(rend_ind);
voc_ng = neg(:); % or using all negative samples
