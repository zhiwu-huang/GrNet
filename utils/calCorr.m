function r_a = calCorr(r)

[n1, n2]=size(r);
I_r = eye(n1);
I_r = I_r(:,1:n2);
[U,S,V]=svd(r'*I_r);
sv=diag(S);
[sv,ind]=sort(sv,'descend');
r_a = sv(1);