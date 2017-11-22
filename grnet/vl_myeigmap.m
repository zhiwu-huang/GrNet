function [Y, R] = vl_myeigmap (R, p, dzdy)
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

%     dzdy = reshape(dzdy,n1,n2,n3,n4);

%     parfor i3 = 1  : n3
    for i3 = 1  : n3

        for i4 = 1 : n4
                U_t = Us(:,:,i3,i4); S_t = Ss(:,:,i3,i4);
                Y(:,:,i3,i4) = calculate_grad_svd(U_t,S_t,p, D,dzdy(:,:,i3,i4));
        end
    end
end

function dzdx = calculate_grad_svd(U,S,p,D,dzdy)
% U = Us{ix}; S = Ss{ix};
diagS = diag(S);
% ind =diagS >(D*eps(max(diagS)));% adding diag!!!
% Dmin = (min(find(ind,1,'last'),D));
Dmin = length(diagS);
ind = 1:Dmin;

% S = S(:,ind); U = U(:,ind);
% dLdC = double(reshape(dzdy,[D D])); dLdC = symmetric(dLdC);

% [max_S, max_I] = max_eig_abs(S,epsilon);
% dLdU = 2*dLdC*U*max_S;
% dLdS = (diag(not(max_I)))*(U'*dLdC*U);
dLdC = zeros(D,D);
A = [ones(D,p) zeros(D,D-p)];
dLdC(logical(A)) = dzdy;
dLdU = dLdC;

if sum(ind) == 1 % diag behaves badly when there is only 1d
    K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))');
    K(eye(size(K,1))>0)=0;
else
%     K = 1./(diag(S).^2*ones(1,Dmin)-(diag(S).^2*ones(1,Dmin))');
    K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))'); % note, problem?

    K(eye(size(K,1))>0)=0;
    K(find(isinf(K)==1))=0; %%%%%!!!!problem?????
    
    
%     if abs(sum(diagS(1:p))-p) < 1e-4
%         K(1:p,1:p) = 0;
%     end
%     if abs(sum(diagS(p+1:end))) < 1e-4
%         K(p+1:end,p+1:end) = 0;
%     end
    

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
%     dzdx = U*(2*K'.*symmetric(U'*dLdU)+dDiag(dLdS))*U';
%     dzdx = U*(2*K'.*symmetric(U'*dLdU))*U';

%     dzdx = U*2*S*symmetric(K'.*(U'*dLdU))*U';

    dzdx = U*symmetric(K'.*(U'*dLdU))*U'; %%%%3.5



end
