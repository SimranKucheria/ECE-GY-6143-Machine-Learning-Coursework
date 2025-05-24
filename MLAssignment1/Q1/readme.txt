ReadMe for Problem 1.

Code files

problem1.m - Main file. Run this to produce all graphs. Iterates through different values of d to find most optimum value. Performs data cross validation by splitting data into train and test datasets.

polyreg.m - File containing function to calculate theta and error for different values of d passed upon function invocation. Also accepts train and test datasets to calculate training and testing errors. 
	Inputs: xTrain, yTrain, Degree of polynomial, xTest, yTest
	Outputs: Training Error, Theta Coefficients, Testing Error
	Implementation: Calculates theta * using the formula pinv(X)*y. Substitutes the theta value into the loss function.

Graphs

LossGraph.png - A graph plotting training/testing loss with respect to different values of d.

PolynomialPlot[0-20].png - Plots for different values of d with the resulting polynomial function plotted.

Dataset
Problem1.mat
