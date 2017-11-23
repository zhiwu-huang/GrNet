function [Y, Y_w] = vl_myfrmap(X, W, dzdy)
%full rank mapping (FRMap) layer

[n1,n2,n3,n4] = size(X);
[n7, n6, n5] = size(W);
Y = zeros(n7,n2,n3,n5);


if nargin < 3    
    for ix = 1  : n3
        for iw = 1 : n5
            if n4 == 1
                Y(:,:,ix,iw) = W(:,:,iw)*X(:,:,ix);
            else
                Y(:,:,ix,iw) = W(:,:,iw)*X(:,:,ix,iw);
            end
        end
    end
else
    Y_w = zeros(n7, n6, n5);
    Y = zeros(n1,n2,n3,n4);
    
    for ix = 1  : n3
        for iw = 1 : n5
            d_t = dzdy(:,:,ix,iw);
            if n4 == 1
                Y(:,:,ix) = Y(:,:,ix)+ W(:,:,iw)'*d_t; 

                Y_w(:,:,iw) = Y_w(:,:,iw)+d_t*X(:,:,ix)';
            else
                Y(:,:,ix,iw) = W(:,:,iw)'*d_t; 

                Y_w(:,:,iw) = Y_w(:,:,iw)+d_t*X(:,:,ix,iw)';
            end
        end
    end
    if n4 == 1
        Y = Y/n5; 
    end
end

