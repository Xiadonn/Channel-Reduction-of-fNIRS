function fitness = maxFitness(feat,label,x)
% change x to logical vector
x = logical(x);
% Make train dataset
train_feat = feat(:,x);
% Standardization
train_feat = zscore(train_feat,0,'all');
% Training SVM Model
[bestacc,~,~] = SVMcgForClass(label,train_feat);
% Number of 0 in x
num_0 = 52 - sum(x);
% Compute fitness
fitness = bestacc + 0.01 * num_0;
end