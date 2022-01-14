function [bestacc,bestc,bestg] = SVMcgForClass(label,train)
%% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(-8:1:8,-8:1:8);
[m,n] = size(X);
cg = zeros(m,n);

eps = 10^(-4);

%% record acc with different c & g, and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;

for i = 1:m
    % Parallel training
    parfor j = 1:n
        cmd = ['-v ',num2str(10),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        cg(i,j) = svmtrain(label,train,cmd);
    end
    
    % Find the bestacc with the smallest c
    for j = 1:n
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
        if abs( cg(i,j) - bestacc )<=eps && bestc > basenum^X(i,j)
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
    end
end