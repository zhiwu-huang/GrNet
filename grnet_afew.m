function [net, info] = grnet_afew(varargin)
%set up the path
confPath;
%parameter setting
opts.dataDir = fullfile('./data/afew') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'grdb_afew_train_gr400_10_int_histeq.mat');
opts.batchSize = 30 ;
opts.test.batchSize = 1;
opts.numEpochs = 500 ;
opts.gpus = [] ;
opts.learningRate = 0.01*ones(1,opts.numEpochs);
opts.weightDecay = 0.0005 ;
opts.continue = 1;
%grnet initialization
net = grnet_init_afew() ;
%loading metadata 
load(opts.imdbPathtrain) ;
%grnet training
[net, info] = grnet_train_afew(net, gr_train, opts);

