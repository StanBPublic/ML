function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
x = X(:,2); % Take everything in second column (fisrt column is ones)

for iter = 1:num_iters    
    h = theta(1) + (theta(2)*x); % hypothesis
    
    % Simultaneously update theta 
    theta_zero = theta(1) - alpha * (1/m) * sum(h-y); % First x = 1
    theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x); % element-wise multiplcation

    theta = [theta_zero; theta_one];
    J_history(iter) = computeCost(X, y, theta);    
end  

disp(min(J_history));
end
