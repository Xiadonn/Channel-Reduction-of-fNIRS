function result = pGAPSO_II(feat,label,opts)
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
N_PSO = 0.2 * N;
N_GA = N - N_PSO;

% Objective function
fun = @maxFitness;
% Number of dimensions
dim = 52;
% Initial
X = Initialization(N,dim);
V = zeros(N_PSO,dim);
% Fitness
fit  = zeros(1,N);
fitG = - inf;
% Evaluate
for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres));
    % Gbest update
    if fit(i) > fitG
        fitG = fit(i);
        Xgb  = X(i,:);
    end
end
% Ranking
[fit,idx] = sort(fit, 'descend');
X         = X(idx,:);
% Partition
fit_PSO = fit(1:N_PSO);
fit_GA  = fit(N_PSO + 1:N);
X_PSO = X(1:N_PSO,:);
X_GA  = X(N_PSO + 1:N,:) > thres;
% Pbest
Xpb  = X_PSO;
fitP = fit_PSO;

%% Pre
curve = - inf;
curve(1) = fitG;
t = 2;

%% Iterations
while t <= max_Iter
    % Part of PSO
    for i = 1:N_PSO
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
    
    % Information exchange: PSO -> GA
    X_comb   = [(X_PSO(:,:) > thres); X_GA];
    fit_comb = [fit_PSO, fit_GA];
    
    % Part of GA
    % Get probability
    prob = fit_comb / sum(fit_comb);
    % Preparation
    Xc1   = zeros(1,dim);
    Xc2   = zeros(1,dim);
    fitC1 = ones(1,1);
    fitC2 = ones(1,1);
    z     = 1;
    for i = 1:N
        if rand() < CR
            % Select two parents
            k1 = RouletteWheelSelection(prob);
            k2 = RouletteWheelSelection(prob);
            % Store parents 
            P1 = X_comb(k1,:);
            P2 = X_comb(k2,:);
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
            fitC2(1,z) = fun(feat,label,Xc2(z,:));
            z = z + 1;
        end
    end
    % Ranking
    Xc   = [X_GA; Xc1; Xc2];
    fitC = [fit_GA,fitC1,fitC2];
    [fitC,idx] = sort(fitC, 'descend');
    % Reserved GA
    fitC = fitC(1:N_GA);
    Xc   = Xc(idx(1:N_GA),:);
    
    % Merge population
    X   = [(X_PSO(:,:) > thres); Xc];
    fit = [fit_PSO,fitC];
    % Ranking
    [fit,idx] = sort(fit, 'descend');
    X         = X(idx,:);
    % Partition
    fit_PSO = fit(1:N_PSO);
    fit_GA  = fit(N_PSO + 1:N);
    X_PSO = X(1:N_PSO,:);
    X_GA  = X(N_PSO + 1:N,:);
    % Gbest update
    if fit(1) > fitG
        fitG = fit(1);
        Xgb  = X(1,:);
    end
    % Pbest update
    for i = 1:N_PSO
        if fit_PSO(i) > fitP(i)
			fitP(i)  = fit_PSO(i);
			Xpb(i,:) = X_PSO(i,:);
        end
    end
    % Save
    curve(t) = fitG;
    fprintf('\nIteration %d Best (pGAPSO-II)= %.2f\n',t,curve(t))
    % Stop the loop if fitG is not updated for 30 iterations
    if t > 30 && t <= max_Iter && abs( curve(t) - curve(t - 30) ) <= 10^(-4)
        break
    end
    t = t + 1;
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

%% Initialization function
function X = Initialization(N,dim)
X = zeros(N,dim);
for i = 1:N
    for d = 1:dim 
        X(i,d) = rand();
    end
end
end