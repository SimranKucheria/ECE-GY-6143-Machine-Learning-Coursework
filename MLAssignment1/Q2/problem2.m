fs2 = load("problem2.mat");
N = size(fs2.x,1);
idx = randperm(N);
PtrainX = fs2.x(idx(1:round(N*0.5)),:);
PtestX = fs2.x(idx(round(N*0.5)+1:end),:);
PtrainY = fs2.y(idx(1:round(N*0.5)),:);
PtestY = fs2.y(idx(round(N*0.5)+1:end),:);
errArr = [];
errTArr = [];
lambdaValues = [0:1000];
for lambda=0:1000
    [TrainErr,theta,TestErr] = polyreglambda(PtrainX,PtrainY,lambda,PtestX,PtestY);
    errArr(end+1) = TrainErr;
    errTArr(end+1) = TestErr;
end   
clf
[M,idx] = min(errTArr);
disp(idx);
disp(M);
plot(lambdaValues,errArr,'b');
hold on
plot(lambdaValues,errTArr,'r');
xlabel(sprintf("Lambda"));
ylabel("Loss");
legend("TrainingLoss","TestingLoss");
print("LambdaLossGraph.png", "-dpng");