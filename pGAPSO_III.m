function result = pGAPSO_III(feat,label,opts)
%% Parameters settings
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end

% Parameters of GA
CR = 0.8;    % crossover rate
MR = 0.01;   % mutation rate
if isfield(opts,'CR'), CR = opts.CR; end
if isfield(opts,'MR'), MR = opts.MR; end

% Parameters of PSO
lb    = 0;
ub    = 1;
thres = 0.5;
c1    = 2;              % cognitive factor
c2    = 2;              % social factor
w     = 0.9;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity

if isfield(opts,'c1'), c1 = opts.c1; end
if isfield(opts,'c2'), c2 = opts.c2; end
if isfield(opts,'w'), w = opts.w; end
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end
if isfield(opts,'thres'), thres = opts.thres; end

% Number of solutions in GA and PSO
halfN = N / 2;

% Objective function
fun = @maxFitness;
% Number of dimensions
dim = 52;
% Initial GA
X_GA  = InitializationGA(halfN,dim);
% Initial PSO
X_PSO = InitializationPSO(halfN,dim);
V = zeros(halfN,dim);
% Fitness
fit_GA  = zeros(1,halfN);
fit_PSO = zeros(1,halfN);
fitG = - inf;

%% Evaluate GA and PSO
% Evaluate GA
for i = 1:halfN
    fit_GA(i) = fun(feat,label,X_GA(i,:));
    % GBest update
    if fit_GA(i) > fitG
        fitG = fit_GA(i);
        Xgb  = X_GA(i,:);
    end
end
% Evaluate PSO
for i = 1:halfN
    fit_PSO(i) = fun(feat,label,(X_PSO(i,:) > thres));
    % GBest update
    if fit_PSO(i) > fitG
        fitG = fit_PSO(i);
        Xgb  = X_PSO(i,:);
    end
end
% PBest
Xpb  = X_PSO;
fitP = fit_PSO;

%% Pre
curve = - inf;
curve(1) = fitG;
t = 2;

%% Iterations
while t <= max_Iter
    % Part of GA
    % Get probability
    prob = fit_GA / sum(fit_GA);
    % Preparation
    Xc1   = zeros(1,dim);
    Xc2   = zeros(1,dim);
    fitC1 = ones(1,1);
    fitC2 = ones(1,1);
    z     = 1;
    for i = 1:halfN
        if rand() < CR
            % Select two parents
            k1 = RouletteWheelSelection(prob);
            k2 = RouletteWheelSelection(prob);
            % Store parents 
            P1 = X_GA(k1,:);
            P2 = X_GA(k2,:);
            % Single point crossover
            ind = randi([1,dim - 1]);
            % Crossover between two parents
            Xc1(z,:) = [P1(1:ind),P2(ind + 1:dim)];
            Xc2(z,:) = [P2(1:ind),P1(ind + 1:dim)];
            % Mutation
            for d = 1:dim
                % First child
                if rand() < MR
                    Xc1(z,d) = 1 - Xc1(z,d);
                end
                % Second child
                if rand() < MR
                    Xc2(z,d) = 1 - Xc2(z,d);
                end
            end
            % Fitness
            fitC1(1,z) = fun(feat,label,Xc1(z,:));
            % GBest update
            if fitC1(1,z) > fitG
                fitG = fitC1(1,z);
                Xgb  = Xc1(z,:);
            end
            fitC2(1,z) = fun(feat,label,Xc2(z,:));
            % GBest update
            if fitC2(1,z) > fitG
                fitG = fitC2(1,z);
                Xgb  = Xc2(z,:);
            end
            z = z + 1;
        end
    end
    
    % Part of PSO
    for i = 1:halfN
        for d = 1:dim
            r1 = rand();
            r2 = rand();
            % Velocity update (2a)
            VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X_PSO(i,d)) + ...
                 c2 * r2 * (Xgb(d) - X_PSO(i,d));
            % Velocity limit
            VB(VB > Vmax) = Vmax; VB(VB < -Vmax) = -Vmax;
            V(i,d) = VB;
            % Position update (2b)
            X_PSO(i,d) = X_PSO(i,d) + V(i,d);
        end
        % Boundary
        XB = X_PSO(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
        X_PSO(i,:) = XB;
        % Fitness
        fit_PSO(i) = fun(feat,label,(X_PSO(i,:) > thres));
		% Pbest update
        if fit_PSO(i) > fitP(i)
			fitP(i)  = fit_PSO(i);
			Xpb(i,:) = X_PSO(i,:);
        end
        % Gbest update
        if fitP(i) > fitG
            fitG = fitP(i);
            Xgb  = Xpb(i,:);
        end
    end
    
    % Merge population
    XX = [X_GA; Xc1; Xc2; (Xpb(:,:) > thres)];
    FF = [fit_GA,fitC1,fitC2,fitP];
    % Select N best solution
    [FF,idx] = sort(FF, 'descend');
    X        = XX(idx(1:N),:);
    fit      = FF(1:N);
    % Save
    curve(t) = fitG;
    fprintf('\nIteration %d Best (pGAPSO-III)= %.2f\n',t,curve(t))
    % Stop the loop if fitG is not updated for 30 iterations
    if t > 30 && t <= max_Iter && abs( curve(t) - curve(t - 30) ) <= 10^(-4)
        break
    end
    t = t + 1;
    
    % Distribute new solutions to GA and PSO randomly
    [X_GA,X_PSO,fit_GA,fit_PSO] = RandomSolutionDistribution(X,fit);
    % Pbest update
    for i = 1:halfN
        if fit_PSO(i) > fitP(i)
            fitP(i)  = fit_PSO(i);
            Xpb(i,:) = X_PSO(i,:);
        end
    end
end

delete(gcp('nocreate'));  % Closing parallel pool

% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1);
% sFeat = feat(:,Sf);
% Store results
result.sf = Sf;
% result.ff = sFeat;
% result.nf = length(Sf);
result.c  = curve;
% result.f  = feat;
% result.l  = label;
end

%% RouletteWheelSelection function
function Index = RouletteWheelSelection(prob)
% Cummulative summation
C = cumsum(prob);
% Random one value, most probability value [0~1]
P = rand();
% Route wheel
for i = 1:length(C)
    if C(i) > P
        Index = i;
        break;
    end
end
end

%% Initialization function of GA
function X = InitializationGA(N,dim)
X = zeros(N,dim);
for i = 1:N
    for d = 1:dim 
        if rand() > 0.5
            X(i,d) = 1;
        end
    end
end
end

%% Initialization function of PSO
function X = InitializationPSO(N,dim)
X = zeros(N,dim);
for i = 1:N
    for d = 1:dim 
        X(i,d) = rand();
    end
end
end

%% RandomSolutionDistribution function
function [X_GA,X_PSO,fit_GA,fit_PSO] = RandomSolutionDistribution(X,fit)
    N       = length(fit);
    idx_PSO = randperm(N,N/2);       % Index array of selected solutions in PSO
    idx_GA  = setdiff(1:N,idx_PSO);  % Index array of selected solutions in GA
    X_GA    = X(idx_GA,:);
    X_PSO   = X(idx_PSO,:);
    fit_GA  = fit(idx_GA);
    fit_PSO = fit(idx_PSO);
end