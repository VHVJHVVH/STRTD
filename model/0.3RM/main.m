s1 = 1 :14;
t1 = 2 :15;
G1 = graph(s1,t1);

A = laplacian(G1);
path = 'tensor.txt';
load X_hat.mat
load T.mat


X = importD(path,15,28,288);
ratio = 0.3;

[Xw,Xnew] = ms_scenario(X,'ms','bm','missing_rate',ratio);
XX= double(X);
W = tenzeros([15,28,288]);
[W1,W2] = find(Xw);
for i = 1:length(W1)
    W(W1(i,1),W1(i,2),W1(i,3))=1;
end


GXY=tensor(X_hat.*(1-W))+Xw;

a=10^2;
b=10^4;
c=10^6;


[B,S,U1,U2,U3] = HOSVD1(GXY,10,25,210);



r1 = 10*25*210;
r2 = 15*10;
r3 = 28*25;
r4 = 288*210;


%m=full(B)
        


L= W.*ttensor(S,{U1,U2,U3}) - W.*tensor(X);

%G = W.*tensor(X) - W.*ttensor(S,{U1,U2,U3});

P = W.*tensor(X);

L1=tenmat(L,1);
L2=tenmat(L,2);
L3=tenmat(L,3);
S1=tenmat(S,1);
S2=tenmat(S,2);
S3=tenmat(S,3);

D_S = ttm(L,{pinv(U1),pinv(U2),pinv(U3)})+a.*S;
D_U1 = double(L1*kron(U3,U2)*S1')+b.*(A*U1+A'*U1)+a.*U1;
D_U2 = double(L2*kron(U3,U1)*S2')+a.*U2;
D_U3 = double(L3*kron(U2,U1)*S3')+a.*U3+c.*(T*T'*U3);

f1 = 0.5 * norm(double(tenmat(L,1)),'fro')^2+0.5 * a* norm(double(tenmat(S,1)),'fro')^2+0.5 * a * norm(double(U1),'fro')^2+0.5 * a * norm(double(U2),'fro')^2+0.5 * a *norm(double(U3),'fro')^2+b*trace(U1'*A*U1)+0.5*c*norm(double(U3'*T),'fro')^2;
f2=f1;
g = [reshape((S.data),[],1);reshape(U1,[],1);reshape(U2,[],1);reshape(U3,[],1)];
g1 = [reshape(D_S.data,[],1);reshape(D_U1,[],1);reshape(D_U2,[],1);reshape(D_U3,[],1)];

E1 = norm(double(tenmat(L,1)),'fro') / norm(double(tenmat(P,1)),'fro');

E2=E1;


step = 0.00000001;
epsilon = sqrt(eps);

for k=1:1500
   
     g = g - step*(1/sqrt(k))*g1;
     
    S = tensor(g(1:r1),[10,25,210]);
    U1 = reshape(g(r1+1:r1+r2),[15,10]);
    U2 = reshape(g(r1+r2+1:r1+r2+r3),[28,25]);
    U3 = reshape(g(r1+r2+r3+1:r1+r2+r3+r4),[288,210]);
    
    L= W.*ttensor(S,{U1,U2,U3}) - W.*tensor(X);
    
    L1=tenmat(L,1);
    L2=tenmat(L,2);
    L3=tenmat(L,3);
    S1=tenmat(S,1);
    S2=tenmat(S,2);
    S3=tenmat(S,3);

    D_S = ttm(L,{pinv(U1),pinv(U2),pinv(U3)})+a.*S;
    D_U1 = double(L1*kron(U3,U2)*S1')+b.*(A*U1+A'*U1)+a.*U1;
    D_U2 = double(L2*kron(U3,U1)*S2')+a.*U2;
    D_U3 = double(L3*kron(U2,U1)*S3')+a.*U3+c.*(T*T'*U3);
    
    g1 = [reshape(D_S.data,[],1);reshape(D_U1,[],1);reshape(D_U2,[],1);reshape(D_U3,[],1)];
    
    f = 0.5 * norm(double(tenmat(L,1)),'fro')^2+0.5 * a* norm(double(tenmat(S,1)),'fro')^2+0.5 * a * norm(double(U1),'fro')^2+0.5 * a * norm(double(U2),'fro')^2+0.5 * a *norm(double(U3),'fro')^2+b*trace(U1'*A*U1)+0.5*c*norm(double(U3'*T),'fro')^2;
    E = norm(double(tenmat(L,1)),'fro') / norm(double(tenmat(P,1)),'fro');
    
    if E<0.02
         break
    end
    ff = f1-f;
    f1=f;
     if E<E1
         E1 = E;
        ki = k;
     end

end

Xtest = double(ttensor(S,{U1,U2,U3}));

WW = tenones(15,28,288);

WW = WW-W;

Sum1 = ((double(X)-Xtest).*double(WW)).^2;

RMSE = sqrt(sum(Sum1(:))/(120960-length(W2)));

Sum5 = (abs((double(X)-Xtest))./double(X)).*double(WW);

MAPE = sum(Sum5(:))/(120960-length(W2));














