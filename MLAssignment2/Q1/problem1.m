fs5 = load("data3.mat");
x = [fs5.data(:,1),fs5.data(:,2),ones(length(fs5.data(:,1)),1)];
y = fs5.data(:,3);
lr = 0.1;
[theta,iter,err,risk] = perceptronsgd(x,y,lr);
disp(iter);
iterVals = 0:iter-1;
clf;
plot(iterVals,err,'r');
hold on
plot(iterVals,risk,'b');
xlabel("Iterations");
ylabel("Error/Risk");
title('Error/Risk graph for lr = 0.1')
legend("Classification Error","Perceptron Error");
print("ErrorRiskGraph3.png", "-dpng");

clf;
ClassAX = [];
ClassBX = [];
for i=1:length(y)
    if y(i)==1
        ClassAX(end+1,:)=x(i,:);
    else    
        ClassBX(end+1,:)=x(i,:);
    end
end    

plot(ClassAX(:,1),ClassAX(:,2),'r.');
hold on
plot(ClassBX(:,1),ClassBX(:,2),'b.');
m=-theta(1)/theta(2); 
c=-theta(3)/theta(2);
y = m*(x(:,1))+c;
plot(x(:,1),y,"b");
title('Classification Boundary for lr = 0.1')
xlabel("X0");
ylabel("X1");

print("ClassificationLine3.png", "-dpng");

