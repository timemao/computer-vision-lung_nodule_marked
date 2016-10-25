function [A]=read3DRaw(rawfileName,M,N,D,dataType)

size=[1 M*N*D];
fid=fopen(rawfileName,'r');
img=fread(fid,size,dataType);
fclose(fid);

B=reshape(img,M*N,D)';
A=zeros(M,N,D);

for i=1:D
   A(:,:,i)=reshape(B(i,:),M,N)'; 
end
