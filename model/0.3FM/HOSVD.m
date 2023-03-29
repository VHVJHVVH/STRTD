
path = 'tensor.txt';
load X_hat.mat
X = importD(path,15,28,288);
ratio = 0.3;
p=0.96
q=0.98
z=0.99
[Xw,Xnew] = ms_scenario(X,'ms','fiber','missing_rate',ratio);
XX= double(X);
W = tenzeros([15,28,288]);
[W1,W2] = find(Xw);
for i = 1:length(W1)
    W(W1(i,1),W1(i,2),W1(i,3))=1;
end


A=tensor(X_hat.*(1-W))+Xw;
u1=double(tenmat(A,1));
u2=double(tenmat(A,2));
u3=double(tenmat(A,3));


[U11,V1,W1]=svd(u1);
[U22,V2,W2]=svd(u2);
[U33,V3,W3]=svd(u3);
X1 = diag(V1);
X2 = diag(V2);
X3 = diag(V3);
for h1=1:length(X1)
   if sum(X1(1:h1))/sum(X1)>p && sum(X1(1:h1-1))/sum(X1) <= p
   R1=h1;
   end
end
R1;
for h2=1:length(X2)
   if sum(X2(1:h2))/sum(X2)>q && sum(X2(1:h2-1))/sum(X2)<=q
    R2=h2;
   end
end
R2;
for h3=1:length(X3)
   if sum(X3(1:h3))/sum(X3)>z && sum(X3(1:h3-1))/sum(X3)<=z
     R3=h3;  
   end
end
R3;   
if R1<2
  R1=R1+1;
end
if R2<2
  R2=R2+1;
end
if R3<2
  R3=R3+1;
end
U1=U11(:,1:R1);
U2=U22(:,1:R2);
U3=U33(:,1:R3);


