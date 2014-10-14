function model = DBC_train(train_data,train_label,nbits, opt1)
% Learning DBC Hyperplanes
%
% Publication:
% Attribute Discovery via Predictable Discriminative Binary Codes. 
%    By M. Rastegari, A. Farhadi, D. A. Forsyth.
%    In Proceeding of ECCV'2012
% 
% Code is writen by Mohammad Rastegari. Report any bugs to mrastega@cs.umd.edu
% 
% Copyright (c) 2012
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the “Software”),  
% THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY.
%
% Usage Syntax:
% 
%  model = DBC_train(train_data, train_label, nbits, opt)
% Input: 
%  train_data: MxN real value matrix : (M: number of dimensions, N: number of samples)
%  train_label: 1xN integer value matrix: N: number of samples
%  nbits: number of DBC hyperplanes
%  opt: SVM options (default: '-B 1 -c 1 -s 1')
%
% Output:
%  model: An structure with two fields "hypothesis" and "nbits"
%
% Thanks to Jonghyun Choi for cleaning the code.  

if ~exist( 'opt1' ), opt1 = []; end
number_of_hypothesis=nbits;

uni_labels=unique(train_label);
num_of_cat=length(uni_labels);

num_exampels_per_cat=sum(train_label==1);
for i=1:num_of_cat
    num_examples_in_cat(i)=length(find(uni_labels(i)==train_label));
end

[m n]=size(train_data);

label_tabel=creating_label_tabel(train_data,train_label,number_of_hypothesis);

for i=1:2
    %% Learning hypothesis(splits)
    hypothesis=train_hypothesis(train_data,label_tabel, opt1);
    if i>1
        for j=1, hypothesis=update_hypothesis(hypothesis,train_data,num_of_cat, opt1); end
    end

    %% Producing binary features
    [m n]=size(train_data);

    binary_features_train = (hypothesis'*[train_data; ones(1,n)])>0;

    % update binary labels
    label_tabel=binary_features_train;
    for j=1:10
        label_tabel=update_label_tabel(label_tabel,num_exampels_per_cat,num_examples_in_cat); 
    end

end

model.hypothesis=hypothesis;
model.nbits=nbits;
