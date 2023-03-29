function X = importD(path,q1,q2,q3)
a = textread(path);

a = reshape(a(1:q1*q2*q3),q1,q2,q3);
X = tensor(a);
    
    