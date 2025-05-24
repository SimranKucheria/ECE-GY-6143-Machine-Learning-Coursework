fs4 = load("dataset4.mat");
x = fs4.X;
y = fs4.Y;
lr = 3;
epsilon = 0.002;
[theta,iter,err,risk] = logisticReg(x,y,lr,epsilon);
disp(iter);
iterVals = 0:iter;
clf;
plot(iterVals,err,'r');
hold on
plot(iterVals,risk,'b');
xlabel("Iterations");
ylabel("Error/Risk");
legend("Error","Risk");
print("ErrorRiskGraph4.png", "-dpng");

clf;

plot(x(:,1),x(:,2),'r.');
hold on
m=-theta(1)/theta(2); 
c=-theta(3)/theta(2);
y = m*(x(:,1))+c;
plot(x(:,1),y,"b");
xlabel("X0");
ylabel("X1");

print("ClassificationLine4.png", "-dpng");

