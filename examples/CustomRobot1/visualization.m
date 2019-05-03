clc; clear all; close all;

load('/home/jeremie/workspaces/universe/build/kalman/data.csv');

% Plot positions
figure;
title('Position Estimate');
plot(data(:,[1,4,7,10]),data(:,[2,5,8,11]),'LineWidth',4);
legend({'True Position', 'Prediction only','EKF Estimate', 'UKF Estimate'});
xlabel('x');
ylabel('y');
return;
% Estimation error (euclidian distance)
figure;
title('Estimate errors from true position (Euclidian distance)');
ekf_error = sqrt((data(:,1)-data(:,7)).^2 + (data(:,2)-data(:,8)).^2);
ukf_error = sqrt((data(:,1)-data(:,10)).^2 + (data(:,2)-data(:,11)).^2);
plot([ekf_error, ukf_error],'LineWidth',4);
legend({'EKF Error', 'UKF Error'});
xlabel('Iteration')
ylabel('Error')
