function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
disp "hi"
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 





% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
disp "hi"
J = (1/(m))*(((-y')*log(sigmoid(X*theta))-((1-y)')*log(1-sigmoid(X*theta))));
J




% =========================================================================

end
