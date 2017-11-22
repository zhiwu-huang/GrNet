function [m_r,i_r] =  maxOrth(r)

m_r = 0; i_r = 1;
for i = 1 : size(r,3)    
    r_a = calCorr(r(:,:,i));
    if r_a > m_r
        m_r = r_a;
        i_r = i;
    end
end

% n_Y = size(r,3);
% dist_Y = zeros(n_Y,n_Y);
% 
% for i = 1 : n_Y
%     Y1 = r(:,:,i)*r(:,:,i)';
%     for j = i+1 : n_Y
%         Y2 = r(:,:,j)*r(:,:,j)';
%         dist_Y(i,j) = Y1(:)'*Y2(:);
%         dist_Y(j,i) = dist_Y(i,j);
%     end
% end
% 
% [m_r,i_r]=sort(sum(dist_Y),'ascend');
% i_r = i_r(1);