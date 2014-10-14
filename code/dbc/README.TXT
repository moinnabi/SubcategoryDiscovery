MATLAB code for Attribute Discovery

 Publication:
 Attribute Discovery via Predictable Discriminative Binary Codes. 
    By M. Rastegari, A. Farhadi, D. A. Forsyth.
    In Proceeding of ECCV'2012
 
 Code is writen by Mohammad Rastegari. Report any bugs to mrastega@cs.umd.edu
 
 Copyright (c) 2012
 Permission is hereby granted, free of charge, to any person obtaining a 
 copy of this software and associated documentation files (the “Software”),  
 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY.


 NOTE: you need to have LIBlinear installed and added it's path to the MATLAB paths.

 You can simply run these two files: 1-"DBC_train.m", 2-"DBC_apply.m"

  >>model=DBC_train(rand(100,1000),[ones(1,500) -ones(1,500)], 32, '-B 1 -c 1 -s 1');
  >>H=DBC_apply(rand(100,500),model);
