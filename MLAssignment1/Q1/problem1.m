fs = load("problem1.mat");
N = size(fs.x,1);
idx = randperm(N);
PtrainX = fs.x(idx(1:round(N*0.5)),:);
PtestX = fs.x(idx(round(N*0.5)+1:end),:);
PtrainY = fs.y(idx(1:round(N*0.5)),:);
PtestY = fs.y(idx(round(N*0.5)+1:end),:);
minErr = 0;
minErrD = 1;
PolynomialX = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];
errTPlot = [];
errPlot = [];
for D=1:20
    [err,model,errT] = polyreg(PtrainX,PtrainY,D,PtestX,PtestY);
    errTPlot(D) = errT;
    errPlot(D) = err;
    
    q  = (min(PtrainX):(max(PtrainX)/300):max(PtrainX))';
    qq = zeros(length(q),D);
    for i=1:D
        qq(:,i) = q.^(D-i);
    end

    clf
    
    plot(PtestX,PtestY,'cO');
    hold on
    plot(q,qq*model,'r')
    xlabel("X");
    ylabel("Y");
    legend("Data","F(X,Theta)");
    title(sprintf("Plot for polynomial with D = %d",D));
    filename = sprintf("PolynomialPlot%d.png",D);
    print(filename, "-dpng");
end   
[M,idx] = min(errTPlot);
disp(idx);
disp(M);
clf
plot(PolynomialX,errTPlot,'b');
hold on
plot(PolynomialX,errPlot,'r');
xlabel(sprintf("P"));
ylabel("Loss");
legend("TestingLoss","TrainingLoss");
print("LossGraph.png", "-dpng");