function [tar far] = Cal_ROC(sim,Mask)

% same_pair_num = sum(sum(Mask,2));
% [num1 num2] = size(sim);
% diff_pair_num = num1*num2 - same_pair_num;

Sim_same = sim(find(Mask==1));
Sim_diff = sim(find(Mask==-1));

same_pair_num = length(Sim_same);
diff_pair_num = length(Sim_diff);

far = [];
tar = [];

min_sim = min(sim(:));
max_sim = max(sim(:));

step = (max_sim - min_sim)/500;
for i =min_sim:step:max_sim
    sim_tar = find(Sim_same>i);
    sim_far = find(Sim_diff>i);
    
    t = size(sim_tar,1)*size(sim_tar,2)/same_pair_num;
    f = size(sim_far,1)*size(sim_far,2)/diff_pair_num;
    
    tar = [tar t];
    far = [far f];
end