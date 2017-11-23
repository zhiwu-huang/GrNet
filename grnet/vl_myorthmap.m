function [Y, R] = vl_myorthmap (R, p, dzdy)
%orthonormal mapping (OrthMap) layer

X = R.x;
A = R.aux;
[n1,n2,n3,n4] = size(X);
if isempty(A) == 1
    Y = zeros(n1,p,n3,n4);

    Us = zeros(n1,n2,n3,n4);
    Ss = zeros(n1,n2,n3,n4);
%     parfor i3 = 1  : n3
    for i3 = 1  : n3
        for i4 = 1 : n4
                X_t = X(:,:,i3,i4);
                [U_t, S_t, V_t] = svd(X_t);
                Us(:,:,i3,i4) = U_t;
                Ss(:,:,i3,i4) = S_t;
                Y(:,:,i3,i4) = U_t(:,1:p);            
        end
    end
    R.aux{1} = Us;
    R.aux{2} = Ss;
else
    Us = A{1};
    Ss = A{2};
    D = size(Ss,2);
    Y = zeros(n1,n2,n3,n4);

%     parfor i3 = 1  : n3
    for i3 = 1  : n3

        for i4 = 1 : n4
                U_t = Us(:,:,i3,i4); S_t = Ss(:,:,i3,i4);
                Y(:,:,i3,i4) = calculate_grad_svd(U_t,S_t,p, D,dzdy(:,:,i3,i4));
        end
    end
end

function dzdx = calculate_grad_svd(U,S,p,D,dzdy)
diagS = diag(S);
Dmin = length(diagS);
ind = 1:Dmin;

dLdC = zeros(D,D);
A = [ones(D,p) zeros(D,D-p)];
dLdC(logical(A)) = dzdy;
dLdU = dLdC;

if sum(ind) == 1 % diag behaves badly when there is only 1d
    K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))');
    K(eye(size(K,1))>0)=0;
else
    K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); 

    K(eye(size(K,1))>0)=0;
    K(find(isinf(K)==1))=0; 
    
    
    ind_s1 = find(abs(diagS-1)<1e-10);
    if isempty(ind_s1) == 0
        K(ind_s1,ind_s1) = 0;
    end

    ind_s0 = find(diagS<1e-10);
    if isempty(ind_s0) == 0
        K(ind_s0,ind_s0) = 0;
    end
    
end
if all(diagS==1)
    dzdx = zeros(D,D);
else
    dzdx = U*symmetric(K'.*(U'*dLdU))*U'; 
end
