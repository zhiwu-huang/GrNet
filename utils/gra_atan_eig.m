function  L = gra_atan_eig(D)
  [M,N] = size(D);
  if isa(D,'gpuArray')
    L = zeros(size(D),'single','gpuArray');
  else
    L = zeros(size(D));
  end
  
  m = min(M,N);
  h1=diag(D);

  L(1:m,1:m) = diag(1./(h1.^2+1));