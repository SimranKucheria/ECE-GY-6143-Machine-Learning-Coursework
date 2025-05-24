function [theta,iter,err,risk] = perceptronsgd(x,y,lr)

theta = rand(1,size(x,2));
iter = 0;
err = [];
risk = [];

    while 1
        i = randi(size(x,1),1);
        pred = x(i,:)*theta';
        diff = lr*x(i,:)*y(i);  
        if y(i)*pred < 0
            theta = theta + diff;
        end
        prediction = x * theta';
        err(iter+1)= getErr(y,prediction,x);
        risk(iter+1)= getRisk(y,prediction,x);
        iter = iter + 1;
        if(err(end)==0)
            break;  
        end
    end
end

function [err] = getErr(y,prediction,x)
    err = 0;
    for i=1:length(y)
        if(step(prediction(i))~=y(i))
            err = err + 1;
        end
    end

    err = err/size(x,1);
end  

function [risk] = getRisk(y,prediction,x)
    risk = 0;
    for i=1:length(y)
            if(step(prediction(i))~=y(i))
                risk = risk - y(i)*prediction(i);
            end    
    end
    risk = risk/size(x,1);
end 

function g = step(z)
    if(z>=0)
        g = 1;
    else
        g = -1;
    end    
end

       
