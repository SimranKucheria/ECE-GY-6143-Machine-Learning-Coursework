ReadMe for Problem 1.

Code files

problem1.m - Main file. Run this with different values of lr to generate the ErrorRisk and ClassificationLines Graphs. Calls the perceptron sgd fn to calculate the number of iterations required to converge for given values of lr along with the error/Risk values. Also returns the coefficients of the classification boundary line.

perceptronsgd.m - File containing function to calculate theta, error/risk and iterations required for different values of lr passed upon function invocation. 
	Inputs: xTrain, yTrain, lr
	Outputs: Theta Coefficients, Iterations required to converge, Error, Risk values across iterations
	Implementation: Calculates theta * X to get the predictions. Initialises theta to random variables and takes a random sample from the dataset, updates theta values with the product of lr * Y(i) *x(i) for misclassified sample. Calculates error using the formula number of incorrectly classified samples/total number of samples. The risk is calculated using -Y(i)*prediction for misclassified samples. Stops updating theta when error is 0.

Graphs

To generate the graphs use the following values of lr and epsilon
1) 10
2) 1
3) 0.1

ClassificationLine[1-3].png - A graph plotting the classification boundary lines between data points for different values of lr and epsilon.

ErrorRiskGraph[1-3].png - Graphs showing the different values of error/risk across the number of iterations.



Dataset
data3.mat
