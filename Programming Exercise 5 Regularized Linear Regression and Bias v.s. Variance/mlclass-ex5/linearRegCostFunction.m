function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
	
%disp(size(X));
%disp(X);
%disp(size(y));
%disp(size(theta));
hypo=X*theta;
regAdjCost= (theta .^2) * lambda;
regAdjCost(1)=0;
squaredError= (hypo - y) .^2;
J = (sum(squaredError(:)) + sum(regAdjCost(:))) /(2*m);


gradTemp=(X'*(hypo - y) ) /m;
gradReg=  lambda/m * theta;
gradReg(1)=0;
grad = gradTemp + gradReg;	

% =========================================================================

grad = grad(:);

end
