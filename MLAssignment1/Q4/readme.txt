ReadMe for Problem 4.

Code files

problem4.m - Main file. Run this with different values of lr and epsilon to generate the ErrorRisk and ClassificationLines Graphs. Calls the logisticReg fn to calculate the number of iterations required to converge for given values of lr and epsilon along with the error/Risk values. Also returns the coefficients of the classification boundary line.

logisticReg.m - File containing function to calculate theta, error/risk and iterations required for different values of lr and epsilon passed upon function invocation. 
	Inputs: xTrain, yTrain, lr, epsilon
	Outputs: Theta Coefficients, Iterations required to converge, Error, Risk values across iterations
	Implementation: Calculates theta * X to get the predictions. Initialises theta to 0 and uses gradient descent to update theta values with the product of lr * derivative of risk function. Calculates error using the formula number of incorrectly classified samples/total number of samples. The risk is calculated using the risk function. Stops updating theta when the norm of the difference between 2 theta values is less than epsilon.

Graphs

To generate the graphs use the following values of lr and epsilon
1) 1,0,001
2) 1,0.003
3) 1,0.005
4) 3,0.002

ClassificationLine[1-4].png - A graph plotting the classification boundary lines between data points for different values of lr and epsilon.

ErrorRiskGraph[1-4].png - Graphs showing the different values of error/risk across the number of iterations.



Dataset
dataset4.mat
