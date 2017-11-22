function [Y, Y_w] = vl_myfc(X, W, dzdy)
%[Y, Y_w] = vl_myfc(X, W, dzdy)
%Fully-connected layer

[n1,n2,n3,n4] = size(X);

X_t = zeros(n1*n2*n4,n3);

for ix = 1 : n3
    x_t = X(:,:,ix,:);
    X_t(:,ix) = x_t(:);
end
if nargin < 3
    Y = W * X_t;
else
    Y = W' * dzdy;
    Y_w = dzdy*X_t';
end