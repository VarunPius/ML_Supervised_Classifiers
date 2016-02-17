%Bayes classification of Randomly generated Gaussian Distribution%

clear all;

%Section i : %
m1=[0 0].';  m2=[2 2].'; Sig=[1 .25; .25 1];

w1 = mvnrnd(m1,Sig,500); 
w2 = mvnrnd(m2,Sig,500);
 
X = [w1.' w2.'];
TrueClass = [ones(1,500) 2*ones(1,500)];

figure;
plot(X(1,1:1000),X(2,1:1000),'.g');

%Section ii :%
figure;
plot(X(1,1:500),X(2,1:500),'.b', X(1,501:1000),X(2,501:1000),'.r');

Pw1 = 0.5; %500 of the 1000 samples have the probability of being in W1%
Pw2 = 0.5; %500 of the 1000 samples have the probability of being in W1%

for i=1:1000
    %p1(i) = pdf('Normal',X(:,i),m1,Sig);%
    %p2(i) = pdf('Normal',X(:,i),m2,Sig);%
    p1(i)=(1/(2*pi*sqrt(det(Sig))))*exp(-(X(:,i)-m1)'*inv(Sig)*(X(:,i)-m1));
    p2(i)=(1/(2*pi*sqrt(det(Sig))))*exp(-(X(:,i)-m2)'*inv(Sig)*(X(:,i)-m2));
end

for i=1:1000
    if(Pw1*p1(i) > Pw2*p2(i))
        class(i)=1;
    else
        class(i)=2;
    end
end

figure;
plot(X(1,class == 1),X(2,class == 1),'.b', X(1,class == 2),X(2,class == 2),'.r');

%Section iii : Error Probability%

Pe = 0;
for i = 1:1000
    if class(i)~=TrueClass(i)
        Pe = Pe + 1;
    end
end

Pep = Pe/1000; %Probability Error Percentage

%Section iv : Risk Minimization Rule Classification%

L=[0 1; .005 0];

for i=1:1000
    if(L(1,2)*Pw1*p1(i) > L(2,1)*Pw2*p2(i))
        LossClass(i)=1;
    else
        LossClass(i)=2;
    end
end

figure; 
plot(X(1,LossClass==1),X(2,LossClass==1),'.b',X(1,LossClass==2),X(2,LossClass==2),'.r');

%Section v : Avg Risk Calculation%

AvgR=0;  %Average risk
for i=1:1000
    if(LossClass(i)~=TrueClass(i))
        if(TrueClass(i)==1)
            AvgR = AvgR + L(1,2);
        else
            AvgR = AvgR + L(2,1);
        end
    end
end
AvgRisk = AvgR/1000;
