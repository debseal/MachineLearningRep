function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=X*theta;
%disp(sigmoid(h));
regAdjCost= (theta .^2) .* (lambda/(2*m));
regAdjCost(1)=0;
tempMatrix = (-y .* log(sigmoid(h)) - (1 - y) .* log (1 - sigmoid(h))) ./ m ;
J = sum (tempMatrix(:)) + sum(regAdjCost(:));
%disp(sum (tempMatrix(:)));
%disp(sum(regAdjCost(:)));
%disp(J);

gradTemp1 =  ((sigmoid(h) .- y) .* X(:,1)) ./m;
grad(1) = sum(gradTemp1(:));
gradTemp2 = ((sigmoid(h) .- y) .* X(:,2)) ./m;
grad(2) = sum(gradTemp2(:)) + + (theta(2,1) * (lambda/m));
gradTemp3 = ((sigmoid(h) .- y) .* X(:,3)) ./m ;
grad(3) = sum(gradTemp3(:)) + (theta(3,1) * (lambda/m));

%disp(grad(1))
%disp(grad(2))
%disp(grad(3))
% =============================================================

end
