function result = sGAPSO(feat,label,opts)
%% Parameters settings
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end

% Parameters of PSO
lb    = 0;
ub    = 1;
thres = 0.5;
c     = 1.49618;
c1    = 2;              % cognitive factor
c2    = 2;              % social factor
w     = 0.7298;         % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity

if isfield(opts,'c'), c = opts.c; end
if isfield(opts,'c1'), c1 = opts.c1; end
if isfield(opts,'c2'), c2 = opts.c2; end
if isfield(opts,'w'), w = opts.w; end
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end
if isfield(opts,'thres'), thres = opts.thres; end

% Parameters of GA
MR = 0.01;   % mutation rate
if isfield(opts,'MR'), MR = opts.MR; end

% Objective function
fun = @maxFitness;
% Number of dimensions
dim = 52;
% Initial population(position),velocity,exemplar and offspring
X = Initialization(N,dim);
V = zeros(N,dim);
E = zeros(N,dim);
O = zeros(N,dim);
% Fitness
fit  = zeros(1,N);
fitE = zeros(1,N);
fitG = - inf;
% Stopping gap of generations
sg = zeros(1,N);

%% Evaluate the initiation
for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres));
    % GBest update
    if fit(i) > fitG
        fitG = fit(i);
        Xgb = X(i,:);
    end
end
% PBest update
Xpb  = X;
fitP = fit;

%% Pre
curve = - inf;
curve(1) = fitG;
t = 2;

%% Iterations
while t <= max_Iter
    % Compute E
    for i = 1:N
        for d = 1:dim
            r1 = rand();
            r2 = rand();
            E(i,d) = (c1 * r1 * Xpb(i,d) + c2 * r2 * Xgb(d)) / (c1 * r1 + c2 * r2);
        end
        % Compute fitness
        fitE(i) = fun(feat,label,(E(i,:) > thres));
    end
    
    % Update
    for i = 1:N
        % Exemplar Update
        for d = 1:dim
            k = randi(N);
            % Crossover
            if fitP(i) > fitP(k)
                r = rand();
                O(i,d) = r * Xpb(i,d) + (1 - r) * Xgb(d);
            else
                O(i,d) = Xpb(k,d);
            end
            % Mutation
            if rand() < MR
                O(i,d) = rand();
            end
        end
        % Compute fitness
        fitO = fun(feat,label,(O(i,:) > thres));
        if fitO > fitE(i)
            E(i,:) = O(i,:);
            sg(i) = 0;
        else
            sg(i) = sg(i) + 1;
        end
        if sg(i) == 7
            sg(i) = 0;
            newidx = TournamentSelection(fitE);
            E(i,:) = E(newidx,:);
        end
        
        % Particle Update
        for d = 1:dim
            r = rand();
            % Velocity update (2a)
            VB = w * V(i,d) + c * r * ( E(i,d) - X(i,d) );
            % Velocity limit
            VB(VB > Vmax) = Vmax; VB(VB < -Vmax) = -Vmax;
            V(i,d) = VB;
            % Position update (2b)
            X(i,d) = X(i,d) + V(i,d);
        end
        % Boundary
        XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
        X(i,:) = XB;
        % Compute fitness
        fit(i) = fun(feat,label,(X(i,:) > thres));
        % PBest update
        if fit(i) > fitP(i)
            fitP(i)  = fit(i);
            Xpb(i,:) = X(i,:);
        end
        % GBest update
        if fitP(i) > fitG
            fitG = fitP(i);
            Xgb  = Xpb(i,:);
        end
    end
    
    % Save
    curve(t) = fitG;
    fprintf('\nIteration %d Best (sGAPSO)= %.2f\n',t,curve(t))
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

function X = Initialization(N,dim)
% Initialize X vectors
X = zeros(N,dim);
for i = 1:N
    for d = 1:dim 
        X(i,d) = rand();
    end
end
end

% 20%M-tournament selection
function bestidx = TournamentSelection(fitE)
bestfit = 0;
bestidx = 0;
N = length(fitE);
newN = 0.2 * N;
index = randperm(N,newN);
for i = 1:newN
    if fitE(index(i)) > bestfit
        bestfit = fitE(index(i));
        bestidx = index(i);
    end
end
end