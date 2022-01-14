clear, clc, close;
%% Prepare dataset
% Load data (Make sure that there is no NaN in dataset)
load('../FeatureData/mean_SZ.mat')
load('../FeatureData/mean_HC.mat')
SZ_feat = mean_SZ;
HC_feat = mean_HC;

% load('../FeatureData/wavelet_SZ.mat')
% load('../FeatureData/wavelet_HC.mat')
% SZ_feat = wavelet_SZ;
% HC_feat = wavelet_HC;

% Generate feature dataset
feat = [SZ_feat; HC_feat];
% Generate label dataset
SZ_label = ones(100,1);
HC_label = zeros(100,1);
label = [SZ_label; HC_label]; 

%% Common parameter settings
opts.N  = 50;     % number of solutions
opts.T  = 200;    % maximum number of iterations

%% Perform channel selection
% type: GA,PSO,pGAPSO_I,pGAPSO_II,pGAPSO_III,sPSOGA,sGAPSO
type = 'GA';
FS = fs(type,feat,label,opts);

%% Prepare results
% Define index of selected features
SC  = sort(FS.sf);  % Selected channels
NSC = length(SC);   % Number of selected channels
% Fitness curve
Fitness = FS.c;
bestFitness = max(Fitness);

%% Print results
fprintf('Best fitness: ');
fprintf('%.2f\n',bestFitness);
fprintf('Best accuracy: ');
fprintf('%.2f\n',bestFitness - 0.01 * (52 - NSC));
fprintf('Selected channels: %d\n',NSC);
for i=1:NSC
    fprintf('%d ',SC(i));
end
fprintf('\n');

%% Plot convergence
plot(Fitness);
grid on;
xlabel('Iterations');
ylabel('Fitness');
title(strrep(type, '_', '\_'));