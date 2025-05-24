ReadMe for Problem 2.

Code files

problem2.m - Main file. Run this to produce the loss graph. Iterates through different values of lambda to find most optimum value. Performs data cross validation by splitting data into train and test datasets.

polyreglambda.m - File containing function to calculate theta and error for different values of lambda passed upon function invocation. Also accepts train and test datasets to calculate training and testing errors. 
	Inputs: xTrain, yTrain, lambda, xTest, yTest
	Outputs: Training Error, Theta Coefficients, Testing Error
	Implementation: Calculates theta * using the formula (X transpose * X + Lambda *I)^-1* X Transpose * Y. Substitutes the theta value into the l2 loss function.

Graphs

LambdaLossGraph.png - A graph plotting training/testing loss with respect to different values of lambda.


Dataset
Problem2.mat
