K=8; J=8; I=8;
dt=.005;
tvec=0:dt:100-dt;
%time vector corresponding to the values in the 
%data vectors
X_vec=randi([-5,5],[1,8]);
Y_mat=randn(J,K);
Z_mat=.05*randn(J,K,I);

x_store=zeros(length(tvec),K);
% y_store=zeros(J,K,length(tvec));
% z_store=zeros(J,K,I,length(tvec));
%range kutta fourth order time step

for i = 1:length(tvec)
[dx1, dy1, dz1] = step(X_vec,Y_mat,Z_mat,K,J,I);

Rx2=X_vec+.5*dt*dx1;
Ry2=Y_mat+.5*dt*dy1;
Rz2=Z_mat+.5*dt*dz1;

[dx2, dy2, dz2] = step(Rx2,Ry2,Rz2,K,J,I);

Rx3=X_vec+.5*dt*dx2;
Ry3=Y_mat+.5*dt*dy2;
Rz3=Z_mat+.5*dt*dz2;

[dx3, dy3, dz3] = step(Rx3,Ry3,Rz3,K,J,I);

Rx4=X_vec+dt*dx3;
Ry4=Y_mat+dt*dy3;
Rz4=Z_mat+dt*dz3;

[dx4, dy4, dz4] = step(Rx4,Ry4,Rz4,K,J,I);

X_vec=X_vec+dt/6*(dx1 + 2*dx2 + 2*dx3 + dx4);
Y_mat=Y_mat+dt/6*(dy1 + 2*dy2 + 2*dy3 + dy4);
Z_mat=Z_mat+dt/6*(dz1 + 2*dz2 + 2*dz3 + dz4);
x_store(i,:)=X_vec;
% Stores the X data. Each row represents a sample 
% of the data 
% y_store(:,:,i)=Y_mat;
% Stores the Y data. Each 2-D matrix represents all
% values at a time. Each column corresponds to 
% y values of each individual x
% z_store(:,:,:,i)=Z_mat;
% Each 3-D array stores the z values at a time. 
% Each column represents z values for an x value
% Each row represents z values for a specific y
%value
end
x_final=(x_store-mean(x_store(:)))/std(x_store(:));
csvwrite('3tier_lorenz_v3.csv',x_store)


function [dx, dy,dz] = step(x_vec,y_mat,z_mat,K,J,I)
xvec=repmat(x_vec,1,3);
ymat=repmat(y_mat,3,1);
zmat=repmat(z_mat,1,1,3);
F=20; c=10; b=10; h=1;
e=10; d=10;

x_minus=xvec(K:(2*K-1));
x_minus2=xvec((K-1):(2*K-2));
x_plus=xvec((K+2):(2*K+1));

y_minus=ymat(J:(2*J-1),:);
y_plus=ymat((J+2):(2*J+1),:);
y_plus2=ymat((J+3):(2*J+2),:);

z_minus=zmat(:,:,I:(2*I-1));
z_minus2=zmat(:,:,(I-1):(2*I-2));
z_plus=zmat(:,:,(I+2):(2*I+1));

y_k=sum(y_mat);

z_kj=sum(z_mat,3);

dx=x_minus.*(x_plus-x_minus2)-x_vec...
    +F-(h*c/b)*y_k;
dy=-c*b*y_plus.*(y_plus2-y_minus)...
    -c*y_mat+(h*c/b)*x_vec-(h*e/d)*z_kj;

dz=e*d*z_minus.*(z_plus-z_minus2)...
    -e*z_mat+(h*e/d)*y_mat;

end