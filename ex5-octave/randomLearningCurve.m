function [error_train, error_val] = ...
    randomLearningCurve(X, y, Xval, yval, lambda)
m = size(X, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

A1=[X y];
A2=[Xval yval];
for i = 1:m
   B1=A1(randperm(m,i),:);
   X_train = B1(:,1:size(X,2));
   y_train = B1(:,size(X,2)+1:size(A1,2));

   B2=A2(randperm(m,i),:);
   Xval = B2(:,1:size(Xval,2));
   yval = B2(:,size(Xval,2)+1:size(A2,2));

   theta = trainLinearReg(X_train,y_train, lambda);
   [J_train, grad] = linearRegCostFunction(X_train, y_train, theta, 0);   % 用lambda =0调用
   [J_val, grad] = linearRegCostFunction(Xval, yval, theta, 0);   % 用lambda =0调用

   error_train(i) =  J_train;
   error_val(i)   =  J_val;

end
