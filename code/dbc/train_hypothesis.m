function hypothesis = train_hypothesis(train_data,label_tabel,opt1)
%
%
%

[m n]=size(label_tabel);
number_of_hypothesis=m;
low_dim=length(train_data(:,1));

% main loop
for i=1:number_of_hypothesis
    train_label=label_tabel(i,:);
        pos_train_idx=find(train_label==1);
        neg_train_idx=find(train_label~=1);
        posneg_train_data=double([train_data(:,pos_train_idx) train_data(:,neg_train_idx)]); % binarized verion of classemes
        posneg_train_label=[ones(length(pos_train_idx),1); -ones(length(neg_train_idx),1)];
        N=length(posneg_train_label);
        Np=sum(posneg_train_label==1);
        Nn=sum(posneg_train_label==-1);
		if isempty( opt1 )
			opt1 = [' -B 1 -c ' num2str(1) ' -s 1 -w-1 ' num2str((1/Nn)) ' -w1 ' num2str(1/Np)];
		end
	
    model = train(posneg_train_label,sparse(posneg_train_data),opt1,'col');
    
    hypothesis(:,i)=model.w';
end