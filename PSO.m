function result = PSO(feat,label,opts)
%% Parameters
lb    = 0;
ub    = 1;
thres = 0.5;
c1    = 2;              % cognitive factor
c2    = 2;              % social factor
w     = 0.9;            % inertia weight
Vmax  = (ub - lb) / 2;  % Maximum velocity

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c1'), c1 = opts.c1; end
if isfield(opts,'c2'), c2 = opts.c2; end
if isfield(opts,'w'), w = opts.w; end
if isfield(opts,'Vmax'), Vmax = opts.Vmax; end
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @maxFitness;
% Number of dimensions
dim = 52;

%% Initial
X = Initialization(N,dim);
V = zeros(N,dim);
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
% PBest
Xpb  = X;
fitP = fit;

%% Pre
curve = - inf;
curve(1) = fitG;
t = 2;

%% Iterations
while t <= max_Iter
    for i = 1:N
        for d = 1:dim
            r1 = rand();
            r2 = rand();
            % Velocity update (2a)
            VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + ...
                 c2 * r2 * (Xgb(d) - X(i,d));
            % Velocity limit
            VB(VB > Vmax) = Vmax; VB(VB < -Vmax) = -Vmax;
            V(i,d) = VB;
            % Position update (2b)
            X(i,d) = X(i,d) + V(i,d);
        end
        % Boundary
        XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
        X(i,:) = XB;
        % Fitness
        fit(i) = fun(feat,label,(X(i,:) > thres));
        % Pbest update
        if fit(i) > fitP(i)
            fitP(i)  = fit(i);
            Xpb(i,:) = X(i,:);
        end
        % Gbest update
        if fitP(i) > fitG
            fitG = fitP(i);
            Xgb  = Xpb(i,:);
        end
    end
    curve(t) = fitG;
    fprintf('\nIteration %d Best (PSO)= %.2f\n',t,curve(t))
    % Stop the loop if fitG is not updated for 30 iterations
    if t > 30 && t <= max_Iter && abs( curve(t) - curve(t - 30) ) <= 10^(-4)
        break
    end
    t = t + 1;
end

delete(gcp('nocreate'));  % Closing parallel pool

%%  Results
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

%% Initialization function
function X = Initialization(N,dim)
X = zeros(N,dim);
for i = 1:N
    for d = 1:dim 
        X(i,d) = rand();
    end
end
end