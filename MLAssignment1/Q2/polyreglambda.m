function [TrainErr,theta,TestErr] = polyreglambda(x,y,lambda,xT,yT)

theta = inv(x'*x + lambda*eye(size(x,2)))*x'*y;

TrainErr   = (1/(2*size(x,1)))* (sum((y-x*theta).^2) + lambda*norm(theta)*norm(theta));
TestErr  = (1/(2*size(xT,1)))* (sum((yT-xT*theta).^2) + lambda*norm(theta)*norm(theta));
