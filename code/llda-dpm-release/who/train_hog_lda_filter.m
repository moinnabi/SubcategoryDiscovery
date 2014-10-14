function [w, b] = train_hog_lda_filter(bg, mu_pos, R, mu_bg)

if ~isa(mu_pos, 'double')
  mu_pos = double(mu_pos);
end

% Remove truncation feature dim
mu_pos = mu_pos(:,:,1:end-1);

[ny nx nf] = size(mu_pos);

if ~exist('R', 'var') || ~exist('mu_bg', 'var')
  [R, mu_bg] = hog_whitening_matrix(bg, nx, ny, true);
end

% compute S^-1*(mu_pos-mu_bg) efficiently
w = R\(R'\(mu_pos(:)-mu_bg));
% compute bias
b = -w'*mu_bg;

%% Add in occlusion feature
%w1 = w;
%w = reshape(w, [ny nx nf]);
%w(:,:,end+1) = 0;
%w = w/norm(w(:));
%subplot(1,2,1);
%visualizeHOG(max(w,0));
%
%s=w1'*feats;
%[s,i]=sort(s, 'descend');
%i=i(1:min(50, length(i)));
%pos=mean(feats(:,i),2);
%w = reshape(R\(R'\(pos-neg)),[ny nx nf]);
%w = w/norm(w(:));
%subplot(1,2,2);
%visualizeHOG(max(w,0));

% Add back truncation feature dimension
w = reshape(w, [ny nx nf]);
w(:,:,end+1) = 0;
