function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];
z2 = X*Theta1';
a2=sigmoid(z2);
a2 = [ones(m,1) a2];
z3= a2*Theta2';
a3 = sigmoid(z3);
hypVec=a3;

yVec=zeros(m,num_labels);
for i=1:m
	yVec(i,y(i))=1;
	%yVec(i,:)=[y(i)==1 y(i)==2 y(i)==3 y(i)==4 y(i)==5 y(i)==6 y(i)==7 y(i)==8 y(i)==9 y(i)==10];
endfor;

for i=1:m
	label_count(i,:)= (-yVec(i,:) .* log (hypVec(i,:))) - ( (1-yVec(i,:)) .* log(1- hypVec(i,:)));
	label_sum(i)=sum(label_count(i,:));
endfor;

JWithoutReg = sum(label_sum(:)) /m;

% Regularizing the cost.
Theta1SqSum=0;
Theta2SqSum=0;

for i=1:hidden_layer_size
	Theta1Sq(i,:)= Theta1(i,:) .^2;
	Theta1Sq(i,1)=0;
	Theta1SqSum += sum(Theta1Sq(i,:));
endfor;

for i=1:num_labels
	Theta2Sq(i,:)= Theta2(i,:) .^2;
	Theta2Sq(i,1)=0;
	Theta2SqSum += sum(Theta2Sq(i,:));
endfor;

J= JWithoutReg + (lambda/(2*m))* (Theta1SqSum + Theta2SqSum) 
%disp(J);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

bDel1 = zeros(size(Theta1));
bDel2 = zeros(size(Theta2));
for t=1:m
	a1B(t,:) = X(t,:);
	z2B(t,:) = a1B(t,:) * Theta1';
	a2B(t,:) = [1 sigmoid(z2B(t,:))];
	z3B(t,:) = a2B(t,:) * Theta2';
	a3B(t,:) = sigmoid(z3B(t,:));
	
	sDel3(t,:)= a3B(t,:) - yVec(t,:);
	sDel2(t,:)= sDel3(t,:)*Theta2(:,2:end) .* sigmoidGradient(z2B(t,:));
	
	bDel1 +=  sDel2(t,:)' * a1B(t,:);
	bDel2 +=  sDel3(t,:)' * a2B(t,:);
endfor;

%disp(size(a1B)); %5000 x 401
%disp(size(a2B)); %5000 x 26
%disp(size(a3B)); %5000 x 10
%disp(size(sDel2)); %5000 x 25
%disp(size(sDel3)); %5000 x 10
%disp(size(bDel1)); %25 x 401
%disp(size(bDel2)); % 10 x 26


Theta1_grad = bDel1/m;
Theta2_grad = bDel2/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

Theta1_grad = bDel1/m;
Theta2_grad = bDel2/m;
Theta1forRegGrad=Theta1;
Theta2forRegGrad=Theta2;
Theta1forRegGrad(:,1)=0;
Theta2forRegGrad(:,1)=0;

Theta1_grad +=  lambda/m * Theta1forRegGrad;
Theta2_grad +=  lambda/m * Theta2forRegGrad;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
