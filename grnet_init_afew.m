function net = grnet_init_afew(varargin)
% grnet_init Initialize a grnet

rng('default');
rng(0) ;

% 2 Blocks (achives the highest accuracy of 34.50% on AFEW)
opts.datadim = [400, 300, 100, 50];
opts.skedim = [8, 8, 8, 8];
opts.pool = [2, 2, 2, 2, 2];
opts.layernum = length(opts.datadim)-2;
Winit = cell(opts.layernum+1,1);


% % 2 Blocks
% opts.datadim = [400, 300, 100, 50];
% opts.skedim = [16, 16, 16, 16];
% opts.pool = [2, 2, 2, 2, 2];
% opts.layernum = length(opts.datadim)-2;
% Winit = cell(opts.layernum+1,1);


% 1 Block
% opts.datadim = [400, 100, 50];
% opts.skedim = [16, 16,16];
% opts.pool = [2, 2,2];
% opts.layernum = length(opts.datadim)-1;
% Winit = cell(opts.layernum+1,1);


for iw = 1 :  opts.layernum
    for i_s = 1 : opts.skedim(iw)
        
        if iw ==1
            A = rand(opts.datadim(iw));
        else
            A = rand(opts.datadim(iw)/2);
        end

        [U1, S1, V1] = svd(A * A');
        Winit{iw}.w(:,:,i_s) = U1(:,1:opts.datadim(iw+1))';
    end
end


f=1/100 ;
classNum = 7;

fdim = opts.datadim(end)*opts.datadim(end)*opts.skedim(end);


theta = f*randn(fdim, classNum, 'single');
Winit{iw+1}.w  = theta';

net.layers = {} ;
net.layers{end+1} = struct('type', 'frmap') ;
net.layers{end}.weight = Winit{1}.w;
net.layers{end+1} = struct('type', 'reorth') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'pooling') ;
net.layers{end}.pool = opts.pool(1);
net.layers{end+1} = struct('type', 'orthmap') ;

net.layers{end+1} = struct('type', 'frmap') ;
net.layers{end}.weight = Winit{2}.w;
net.layers{end+1} = struct('type', 'reorth') ;
net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'pooling') ;
net.layers{end}.pool = opts.pool(2);
net.layers{end+1} = struct('type', 'orthmap') ;

net.layers{end+1} = struct('type', 'projmap') ;
net.layers{end+1} = struct('type', 'fc');
net.layers{end}.weight = Winit{end}.w;
net.layers{end+1} = struct('type', 'softmaxloss') ;
