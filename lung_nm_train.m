% lung_nodule_marked
function [net, info] = lung_nm_train(varargin)
% CNN_lung_nm_train  Demonstrated MatConNet on lung_nodule_marked
% 1 matconvnet setup
matconvnet_dir='H:\dataset\matconvnet-1.0-beta8\matconvnet-1.0-beta8\matlab';
run(fullfile(matconvnet_dir,'vl_setupnn'));
% 2 information for data and others
% data-dir: datadir='H:\肺结节2\category';
opts.dataDir = fullfile('data') ;
opts.expDir = fullfile('data','lung-nm-baseline') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 20 ;  % ？？ 100           % to change
opts.train.numEpochs = 1 ;   % 迭代代数  100     % to change
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = 0.001 ;  %学习速率
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
  imdb=imdb.imdb;
else
  imdb = load('imdb.mat') ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Define a network similar to LeNet
f=1/100 ;
net.layers = {} ;
% conv1 98*98-->92*92 follow below
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(7,7,12,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
% pool 1 92--46
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
% conv2 46--40
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(7,7,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
% pool 2  40--20
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
% conv 3  20--14
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(7,7,50,200, 'single'),...
                           'biases', zeros(1,200,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
% ------------------------ pool3 & conv4 my_add -----------------------                                       
% pool 3 14--7                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;     
% conv 4  7-1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(7,7,200,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;   
% ------------------------ pool3 & conv4 my_add -----------------------                        
% relu  
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,4, 'single'),...
                           'biases', zeros(1,4,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
% softmax_loss
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                 Train
% --------------------------------------------------------------------

% Take the mean out and make GPU if needed
imdb.images.data = bsxfun(@minus, imdb.images.data, mean(imdb.images.data,4)) ;
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
% function imdb = getLung_nm_Imdb(opts)
% % --------------------------------------------------------------------
% files = {'Infiltration', ...
%          'no_Infiltration', ...
%          'orginal_adenocarcinoma', ...
%          'others'} ;
% 
% mkdir(opts.dataDir) ;
% 
% 
% 
% % train1 n1=100;n2=50;
% n1=60e3;n2=10e3;
% f=fopen(fullfile(opts.dataDir,files{1}, 'train-images-idx3-ubyte'),'r') ;
% x1=fread(f,inf,'uint8');
% fclose(f);
% x1=x1(17:(n1*28*28+16));
% x1=permute(reshape(x1,28,28,n1),[2 1 3]) ;  %change
% 
% f=fopen(fullfile(opts.dataDir, files{3},'t10k-images-idx3-ubyte'),'r') ;
% x2=fread(f,inf,'uint8');
% fclose(f) ;
% x2=x2(17:(n2*28*28+16));
% x2=permute(reshape(x2,28,28,n2),[2 1 3]) ;  % change
% 
% f=fopen(fullfile(opts.dataDir, files{2},'train-labels-idx1-ubyte'),'r') ;
% y1=fread(f,inf,'uint8');
% fclose(f) ;
% y1=y1(9:n1+8);
% y1=double(y1')+1 ;
% 
% f=fopen(fullfile(opts.dataDir, files{4},'t10k-labels-idx1-ubyte'),'r') ;
% y2=fread(f,inf,'uint8');
% fclose(f) ;
% y2=y2(9:n2+8);
% y2=double(y2')+1 ;
% 
% imdb.images.data = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
% imdb.images.labels = cat(2, y1, y2) ;
% imdb.images.set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
% imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
