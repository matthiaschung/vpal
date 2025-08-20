%Test case: Denoising
%This script solves the denoising problem argmin_x ||Ix - b|_2^2 + ||Dx||_1 with D the finite
%difference opterator.  

%Generate noisy image
img = phantom(64);
xtrue = img;
b = imnoise(img, 'gaussian', 0, 0.01);  
b = b(:);

%regularization parameters
eta = 1;
mu = .09;

%%
%vpal with (gradient step direction)
D = dOperator('finite difference', [64 64]);
options = {'D', D,'mu', mu ,'display','iter', 'maxIter', 1000};
x = vpal(1, b(:), options);  

%plot results:
subplot(1, 3, 1)
imshow(xtrue)
hold on
title("xtrue")
hold off
subplot(1, 3, 2)
imshow(reshape(x, [64, 64]))
title("VPALgradient reconstruction")
subplot(1, 3, 3)
imshow(reshape(b, [64, 64]))
title("Noisy Image")


%%
%vpalNL reconstruction (ncg step direction)

%set up constrained optimization problem
q = @(x) 0.5*norm(x - b, 2)^2;
gradq = @(x) x - b;
D = dOperator('finite difference', [64 64]);
r = @(y) mu*norm(y, 1);
R = -eye(size(D, 1));
Q = D;
v = zeros(size(D, 1), 1);

%linearized stepsize:
stepsize = @(x, y, c, eta, s) ((b'*s - x'*s) + (eta^2)*(-1*(s'*(Q'*(R*y))) + s'*(Q'*v) - s'*(Q'*c)))/(s'*s + s'*(Q'*(Q*s)));

options = {'eta', eta, 'display', 'iter', 'maxIter', 1000, 'jmax', 2, 'stepsize', stepsize};

%define Z(x, c)
Z = @(x, c) sign(D*x + c).*max(abs(D*x + c) - mu/(eta^2), 0);

[x, info] = vpalnl(q, r, gradq, Z, R, Q, v, options);

%plot results
subplot(1, 3, 1)
imshow(xtrue)
hold on
title("xtrue")
hold off
subplot(1, 3, 2)
imshow(reshape(x, [64, 64]))
title("VPALnl reconstruction")
subplot(1, 3, 3)
imshow(reshape(b, [64, 64]))
title("Noisy Image")