function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %   gradientDescent([1 5; 1 2; 1 4; 1 5], [1 6 4 2] ', [0 0]',0.01, 1000 )
        H=(X*theta);
        S1=sum(H-y);;
        J1=S1/(m);

        S2=sum((H-y).*X(:,2));
        J2=S2/(m);
        theta(1,1)=theta(1,1)-alpha*J1
        theta(2,1)=theta(2,1)-alpha*J2




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
