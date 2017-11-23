function [Y, R] = vl_mypooling(R, pool, dzdy)
%projection pooling (ProjPooling) layers

X = R.x;
A = R.aux;

[n1,n2,n3,n4] = size(X);
Y = zeros(n1/pool,n2/pool,n3,n4);
IY = zeros(n1,n2,n3,n4);
tI = zeros(pool,pool,n3,n4);
tI(1,1,:,:) = 1;

if isempty(A)==1
    for ix = 1 : pool : n1
        for iy = 1 : pool : n2
            r_tt = X(ix:ix+(pool-1),iy:iy+(pool-1),:, :);            
            r_tt = reshape(r_tt,[pool*pool n3 n4]);             
            r_mm = mean(r_tt,1);
           
            Y(floor(ix/pool)+1,floor(iy/pool)+1,:,:) =  r_mm;
            
            IY(ix:ix+(pool-1),iy:iy+(pool-1),:,:) = tI;
            
        end
    end
    R.aux = IY;

else
    Y = zeros(n1,n2,n3,n4);
    Y(logical(A)) = dzdy/(pool^2);
end

