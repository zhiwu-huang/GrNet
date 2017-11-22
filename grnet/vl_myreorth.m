function [Y,R] = vl_myreorth(R, dzdy)
X = R.x;

[n1,n2,n3,n4] = size(X);


Y = zeros(n1,n2,n3,n4);

if isempty(R.aux)==1
    Qs = zeros(n1,n2,n3,n4);
    Rs = zeros(n2,n2,n3,n4);
    for ix = 1  : n3
        if n4 == 1
            [Qs(:,:,ix),Rs(:,:,ix)] = qr(X(:,:,ix),0); 
            Y(:,:,ix) = Qs(:,:,ix);
        else
            for iy = 1 : n4
                [Qs(:,:,ix,iy),Rs(:,:,ix,iy)] = qr(X(:,:,ix,iy),0);

                Y(:,:,ix,iy) = Qs(:,:,ix,iy);
            end
        end
    end
    R.aux{1} = Qs;
    R.aux{2} = Rs;
else
    Qs = R.aux{1};
    Rs = R.aux{2};
    for ix = 1  : n3
        if n4 == 1
            Q = Qs(:,:,ix); R = Rs(:,:,ix);
            T = dzdy(:,:,ix);
            dzdx = Compute_Gradient_QR(Q,R,T);
            Y(:,:,ix) =  dzdx;
        else
            for iy = 1 : n4
                Q = Qs(:,:,ix,iy); R = Rs(:,:,ix,iy);
                T = dzdy(:,:,ix,iy);
                dzdx = Compute_Gradient_QR(Q,R,T);
                Y(:,:,ix,iy) =  dzdx;
            end            
        end
    end
end

function dzdx = Compute_Gradient_QR(Q,R,T)
m = size(Q,1);
dLdC = double(T);
dLdQ = dLdC;

S = eye(m)-Q*Q';
dzdx_t0 = Q'*dLdQ;
dzdx_t1 = tril(dzdx_t0,-1);
dzdx_t2 = tril(dzdx_t0',-1);
dzdx = (S'*dLdQ+Q*(dzdx_t1-dzdx_t2))*(inv(R))';
