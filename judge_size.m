% 10/16
clear;clc

filepath='H:\肺结节2\other';   %浸润性 微浸润性  原位腺癌  other
x=textread(fullfile(filepath,'LIST.TXT'),'%s');
n=length(x);
x=x(1:n-2);
for i=1:n-2
    images=[];
    images=load(fullfile(filepath,x{i}));  % {} not ()
    images=images.B;
    size1=size(images{1,1});
    if (length(size1)==2)
        size2=1;
        sizeof_images(i,:)=[size1,size2];
    else
        sizeof_images(i,:)=size1;
    end
  %  x{i}=[x{i},sizeof_images(i,:)];
end
% x=[x;sizeof_images];
 m=max(sizeof_images,[],1)
 find(sizeof_images(:,3)>10) % layers>=10
 find(sizeof_images(:,2)>98)
 hist(sizeof_images(:,3))
