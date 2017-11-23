function Y = vl_myprojmap(X, dzdy)
%projection mapping(ProjMap) layer

[n1,n2,n3,n4] = size(X);

Y = zeros(n1,n1,n3,n4);

if nargin < 2
    for ix = 1: n3
        if n4 == 1
            Y(:,:,ix) = X(:,:,ix)*X(:,:,ix)';
        else
            for iy = 1 : n4
                Y(:,:,ix,iy) = X(:,:,ix,iy)*X(:,:,ix,iy)';
            end
        end
    end
else
    Y = zeros(n1,n2,n3,n4);
    [n5,n6,n7,n8] = size(dzdy);
    
    
    for ix = 1: n3
        d_t = dzdy(:,ix);    
       
        
        if n7 ==1
            
            if n4 ==1
                d_t = reshape(d_t,[n1 n1]);
                Y(:,:,ix) = 2*d_t*X(:,:,ix);
            else
                d_t = reshape(d_t,[n1 n1 n4]);
                
                for iy = 1 : n4
                    Y(:,:,ix,iy) = 2*d_t(:,:,iy)*X(:,:,ix,iy);
                end
            end
        else
            for iy = 1 : n4
                Y(:,:,ix,iy) = 2*dzdy(:,:,ix,iy)*X(:,:,ix,iy);
            end
        end

    end
end
