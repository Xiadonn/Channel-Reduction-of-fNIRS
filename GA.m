function result = GA(feat,label,opts)
%% Parameters
CR = 0.8;    % crossover rate
MR = 0.01;   % mutation rate

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'CR'), CR = opts.CR; end
if isfield(opts,'MR'), MR = opts.MR; end

% Objective function
fun = @maxFitness;
% Number of dimensions
dim = 52;

%% Initial
X = Initialization(N,dim);
% Fitness
fit  = zeros(1,N);
fitG = - inf;
% Evaluate
for i = 1:N
    fit(i) = fun(feat,label,X(i,:));
    % Best update
    if fit(i) > fitG
        fitG = fit(i);
        Xgb  = X(i,:);
    end
end

%% Pre
curve = - inf;
curve(1) = fitG;
t = 2;

%% Iterations
while t <= max_Iter
    % Get probability
    prob = fit / sum(fit);
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
            P1 = X(k1,:);
            P2 = X(k2,:);
            % Single point crossover
            ind = randi([1, dim - 1]);
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
    % Merge population
    XX = [X; Xc1; Xc2];
    FF = [fit,fitC1,fitC2];
    % Select N best solution
    [FF,idx] = sort(FF, 'descend');
    X        = XX(idx(1:N),:);
    fit      = FF(1:N);
    % Best agent
    if fit(1) > fitG
        fitG = fit(1);
        Xgb  = X(1,:);
    end
    % Save
    curve(t) = fitG;
    fprintf('\nGeneration %d Best (GA)= %.2f\n',t,curve(t))
    % Stop the loop if fitG is not updated for 30 iterations
    if t > 30 && t <= max_Iter && abs( curve(t) - curve(t - 30) ) <= 10^(-4)
        break
    end
    t = t + 1;
end

delete(gcp('nocreate'));  % Closing parallel pool

%% Results
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos(Xgb == 1);
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
        if rand() > 0.5
            X(i,d) = 1;
        end
    end
end
end