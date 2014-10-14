function H=DBC_apply(data,model)
% Testing DBC Hyperplanes
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
%  H=DBC_apply(data,model)
% Input: 
%  data: MxN: (M: number of dimensions, N: number of samples)
%  model: Output of DBC_train function
%
% Output:
%  H: nbitsxN
%
% Thanks to Jonghyun Choi for cleaning the code.  

[m n]=size(data);

H=(model.hypothesis'*[data; ones(1,n)])>0;
