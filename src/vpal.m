function [x, f, info] = vpal(A, b, options)
% function [x, f, info] = vpal(A, b, options)
%
% Authors:
%   (c) Matthias Chung (e-mail: matthias.chung@emory.edu) in October 2022
%       Rosemary Renaut
%
% MATLAB Version: 9.10.0.1649659 (R2021a) Update 1
% 
% Version:
%   Tia: 04/04/23 new feature: option of using box constraints
%   Tia: 02/14/23 new feature: selection of step size (linearized and
%   optimal)
%   Tia: 09/27/22 new feature: inital guess x now available
%   Tia: update for distribution 09/21/22
%   Tia: changes to stopping criteria 10/13/21
%   RR3: Version June 18 edited to merge FSB version of Tia downloaded June 18
%   Tia: updating version 6/18
%   RR: Updating June 23
%   RR: Updating June 30
%   RR: Updating July 19 for finding lambda (discrepancy) or mu (dof)
%
% Description:
%   Variable projected augmented Lagrangian method for solving l_1
%   regularized problems
%
%        x = argmin_x  f(x) = 1/2||Ax-b||^2 + mu ||Dx||_1
%
%   where D is the convolution operator, mu > 0 (optional s.t. xmin <= x <= xmax)
%
% Input arguments:
%   A           - forward model (m x n) may be an object
%                 if A is scalar or empty, method is denoising
%   b           - observation vector (m x 1) (column vector)
%   options       [further options of algorithm] (needs to include
%                   bisection options)
%     .x             - inital guess of x [default x = 0]
%     .mu            - regularization parameter
%     .lambda        - Lagrange multiplier - only need squared
%     .sigma         - standard deviation of noise in data. (for finding mu)
%     .tol           - tolerance [ 1e-6 ]
%     .D             - dOperator object in regularization term ||Dx||_1 otherwise
%                      default D is finite differences in 2D (other options Laplace
%                      operater, identity, or 3D operator)
%     .stepSize      - [ {'linearized'} | 'optimal' ]
%     .bnd           - box constraints (default unconstrained), options
%                      interval [xmin, xmax]. If xmin, xmax scalars, then uniform bounds. 
%                      If xmin = -Inf or xmax = Inf just one sided bounds are considered. 
%                      If xmin, xmax column vectors of length n bounds are considered as non-uniform bounds.
%     .dof           - degree of freedom parameter for early stopping of
%                      ill-conditioned problems ||r||^2 < dof + sqrt(2dof)
%     .maxIter       - maximal number of iterations [ 10 * length(b) ]
%     .display       - print to display [ {'off'} | 'iter' | 'final' ]
%
% Output arguments:
%   x           - local minimizer
%   f           - normalized loss (Ax-b)/norm(b)
% %   chi2        - chi2 function(mu) RR7 - is needed external for est
% %   bisect_iter - number of iterations to find optimal mu
%   iter        - number of SB iterations used -needed for sanity check  RR8 remove later
%   info          [additional info on algorithm]
%      .iter    - number of iterations
%      .f       - function value
%      .tol     - selected tolerance
%      .maxIter - selected maximum number of iterations
%      .mu      - regularization parameter during iteration
%      .lambda  - Lagrange multiplier during iteration lambda^2
%      .stop    - stopping criteria during iteration (loss, x, maxIter)
%      .chi2    - the dof test
%      .normr   - residual norm
%      .relerr  - may remove and always assume no x provided RR?
%      .tv      - norm(Dx,1)
%      .D       - d-operator
%
% Example:
%
% Reference: M Chung and R Renaut. The Variable Projected Augmented Lagrangian Method, ArXiv preprint:2207.08216 (https://arxiv.org/abs/2207.08216), 2022.
%

info.version = '2023/04/04.0';

if nargin < 1, fprintf('  Current vpal version: %s\n',info.version); return, end

% initialize default input options
maxIter = 10*size(b,1); display = 'off'; tol = 1e-6; xtrue = [];
lambda = 1; mu = 0; getAllInfo = 0; stepSize = 'linearized'; dof = 0;

if nargin == nargin(mfilename) % rewrite default parameters if needed
  for j = 1:2:length(options)
    eval([options{j},'= options{j+1};'])
  end
end

if nargout > 2, getInfo = 1; else, getInfo = 0; end

if getInfo % general info of method
  info.tol      = tol;
  info.maxIter  = maxIter;
end

% display and algorithm info
if strcmp(display, 'iter') || strcmp(display,'final')
  fprintf('\nvpal algorithm (c) Matthias Chung & Rosemary Renaut, June 2021\n');
  if strcmp(display,'final') == 0
    fprintf('\n %-5s %-6s %-14s \n','iter','loss','stop criteria');
  end
end

if max(size(A)) == 1; n_A = size(b,1); else, n_A = size(A,2); end % get number of unknowns

if exist('bnd','var')           % check box constraints
    if numel(bnd) == 2 % uniform case
        if bnd(2) == inf        % lower bound case
            bound_case = 1; 
        elseif bnd(1) == -inf   % upper bound case
            bound_case = 2; 
        else                    % uniform lower-upper bound case
            bound_case = 3; 
        end
    elseif numel(bnd) == 2*n_A  % nonuniform lower upper bound case
        bound_case = 4;
    else
        error('box constraints not properly set.')
    end
else 
    bound_case = 0; % no box constraints set
end

if ~exist('D','var') % no operator is provided, default is identity
  D = dOperator('identity',n_A);
elseif isa(D,'dOperator') % operator is provided and of class dOperator
    m_D = D.sizes(1);
else
    m_D = size(D,1); % operator is provided and is matrix
end


%edit
c = sparse(m_D,1); y = c; normb = norm(b); f = inf; xOld = inf;             % initialize
iter = 1; lambda2 = lambda^2;
if exist('x','var')
    Dx = D*x;  r = A*x-b;
else 
    Dx = c; x = zeros(n_A,1); r = -b; % here Dx = 0 and r = -b since x = 0
end
 
while 1

  % step 1 Tikonov CG update
  g = A'*r + lambda2 *(D'*(Dx - (y-c)));
  Ag = A*g; Dg = D*g;

  switch stepSize
      case 'linearized'
        alpha = (g'*g)/((Ag'*Ag) + lambda2*(Dg'*Dg));
      case 'optimal'
        gAAg = Ag'*Ag;
        objFcn_alpha = @(alpha) fcn_alpha(alpha, Dx+c, 0.5*gAAg, Dg, r'*Ag, mu/lambda2, lambda2, mu);
        alpha0 = (g'*g)/(gAAg + lambda2*(Dg'*Dg));
        alpha = fminsearch(objFcn_alpha, alpha0);
  end

  x = x - alpha*g;
  
  % project onto feasible region
  switch bound_case
      case 1 % lower bound case
          x(x<bnd(1)) = bnd(1);
      case 2 % upper bound case
          x(x>bnd(2)) = bnd(2);
      case 3 % uniform lower-upper bound case
          x(x<bnd(1)) = bnd(1); x(x>bnd(2)) = bnd(2);
      case 4 % nonuniform lower-upper bound case
          x(x<bnd(:,1)) = bnd(x<bnd(:,1),1); x(x>bnd(:,2)) = bnd(x>bnd(:,2),2);
  end

  % update Dx and residual
  r = A*x - b; Dx = D*x;

  % step 2 shrinkage
  c = Dx + c; y = sign(c).*max(abs(c) -  mu/lambda2, 0);

  % step 3 update c
  c = c - y;

  % calculate loss and tv estimate
  fOld = f; f = 0.5*norm(r)^2 + mu*norm(Dx,1);

  % stopping criteria
  stop1 = abs(fOld - f)  <= tol * (1 + f);
  stop2 = norm(xOld - x,'inf') <= sqrt(tol) * (1 + norm(x,'inf'));
  stop3 = iter > maxIter-1;
  stop4 = norm(r)^2 < dof + sqrt(2*dof); % stopping criteria for ill posed systems using degree of freedom argument

  if getInfo
    info.loss(iter)     = norm(r)/normb;
    info.f(iter)        = f;
    info.alpha(iter)    = alpha;
    info.stop(:,iter)   = [stop1;stop2;stop3];
    if ~isempty(xtrue), info.relerr(iter) = norm(x(:)-xtrue(:))/norm(xtrue(:)); end
    if getAllInfo
      info.x(:,iter)      = x;
      info.g(:,iter)      = g;
      info.c(:,iter)      = c;
      info.y(:,iter)      = y;
      info.Dx(:,iter)     = Dx;
      info.ADg(:,iter)    = [Ag;Dg];
      info.r(:,iter)      = r;
    end
  end

  if strcmp(display,'iter') % display iteration results
    fprintf('%5d %14.6e %4d%1d%1d%1d\n', iter, f, stop1, stop2, stop3, stop4);
  end

  if (stop1 && stop2) || (stop3 || stop4) % check stop criteria
    if stop3
      warning('Matlab:vpal:maxIter',...
        'Maximum number of iterations reached. Return with recent values.')
    end
    if stop4
      warning('Matlab:vpal:dof',...
        'Degree of freedom argument reached. Return with recent values.')
    end    
    break;
  end

  xOld = x; iter = iter + 1;

end

if (strcmp(display,'iter') || strcmp(display,'final')) && ~stop3 % display
  fprintf('\nLocal minimizer found. Function value is %1.8e.\n', f);
end

if getInfo % parameters used for the estimation
  info.tv       = norm(Dx,1);
  info.normr    = norm(r);
  info.chi2     = info.normr^2 + mu*info.tv;%% RR13 note normr calculated for chi
  info.lambda   = sqrt(lambda2);
  info.mu       = mu;
  info.tol      = tol;
  info.maxIter  = maxIter;
  info.iter     = iter;
  info.operator = D;
end

end

% objective function for optimal step size alpha
function f_proj = fcn_alpha(alpha, Dxc, gAAg_hf, Dg, rAg, mulambda2, lambda2, mu)
DxcalphaDg = Dxc - alpha*Dg;
Z = sign(DxcalphaDg).*( max(abs(DxcalphaDg) - mulambda2,0) );
f_proj = alpha^2*gAAg_hf -alpha*rAg + 0.5*lambda2*norm(DxcalphaDg - Z)^2 + mu*norm(Z,1);
end