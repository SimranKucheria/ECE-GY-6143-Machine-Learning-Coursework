function [theta,iter,err,risk] = logisticReg(x,y,lr,epsilon)

theta = zeros(size(x,2),1);
iter = 0;
err = [];
risk = [];

while 1
    fx = x*theta;
    prediction = sigmoid(fx); 
    diff = lr*(x.'*(y-prediction))/size(x,1);
    err(iter+1)= getErr(y,prediction,x);
    risk(iter+1)= getRisk(y,prediction,x);
    if norm(diff)<epsilon
        break
    else
        theta = theta + diff;
        iter = iter + 1;
    end
end

function [err] = getErr(y,pred,x)
    err = 0;
    for i=1:length(y)
        if(pred(i)>0.5~=y(i))
            err = err + 1;
        end
    end

    err = err/size(x,1);
end  

function [risk] = getRisk(y,pred,x)
    risk = 0;
    for i=1:length(y)
            if y(i)==1
                risk = risk - log(pred(i));
            else    
                risk = risk - log(1-pred(i));
            end    
        end
    end
    risk = risk/size(x,1);
end 

function g = sigmoid(z)
    g  = zeros(length(z),1);
    for i = 1:length(z)
        g(i) = 1/(1 + exp(-z(i)));
    end
end

       
