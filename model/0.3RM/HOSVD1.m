function [B,S,U1,U2,U3] = HOSVD1(A,r1,r2,r3)

% U1 = nvecs(A,1,r1); %<-- Mode 1
% U2 = nvecs(A,,r2); %<-- Mode 2
% U3 = nvecs(A,3,r3); %<-- Mode 3
% S = ttm(A,{pinv(U1),pinv(U2),pinv(U3)}); %<-- Core
% B = ttensor(S,{U1,U2,U3}); %<-- HOSVD of X


u1=double(tenmat(A,1));
u2=double(tenmat(A,2));
u3=double(tenmat(A,3));


[U1,V1,W1]=svds(u1,r1);
[U2,V2,W2]=svds(u2,r2);
[U3,V3,W3]=svds(u3,r3);

% U1=U1(:,1:r1);
% U2=U2(:,1:r2);
% U3=U3(:,1:r3);

S = ttm(A,{U1',U2',U3'});
B = ttm(S,{U1,U2,U3});
%A=ttm(S,{U1,U2,U3})
% Z1=full(X);
% Z=full(Y);
% a=norm(full(Y) - X);

