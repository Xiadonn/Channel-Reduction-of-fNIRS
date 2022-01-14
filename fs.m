function model = fs(type,feat,label,opts)
switch type
    case 'GA'         ; fun = @GA;
    case 'PSO'        ; fun = @PSO;
    case 'pGAPSO_I'   ; fun = @pGAPSO_I;
    case 'pGAPSO_II'  ; fun = @pGAPSO_II;
    case 'pGAPSO_III' ; fun = @pGAPSO_III;
    case 'sPSOGA'     ; fun = @sPSOGA;
    case 'sGAPSO'     ; fun = @sGAPSO;
end
tic;
% Enabling parallel pool
parpool(6);
% Run
model = fun(feat,label,opts);
% Computational time
t = toc;
model.t = t;
fprintf('\nProcessing Time (hours): %.2f\n',t/3600);
end