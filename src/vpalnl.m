
%Authors: Jack Michael Solomon (email: jasolo3@emory.edu), Matthias Chung, Rosemary Renaut

%This code implements the most general case of the vpal algorithm with a non-linear conjugate gradient search direction.
%We assume a smooth forward operator, with a convex, potentially non-smooth
%regularization term, i.e.
% argmin_(x, y) f(x, y) = q(x) + r(y), with q(x) smooth, r(y) convex s.t. 
% Qx + Ry = v, where Q and R are matracies of appropriate sizes.  For
% further details, see DOI 10.1088/1361-6420/addde4

%The augmented lagrange is denote h(x, y; c) and is given by
%h(x, y; c) = q(x) + r(y) + (\eta^2/2)(||Qx + Ry - v + c||_2^2 - ||c||_2^2)

%Note that for ths implementation it is assumed that regularization
%paramaters are passed in as part of q, r (except for Lagrangian penalty
%parameters).

%Inputs:

%q(x), r(y): objective functions from separable f(x, y)
%Q, R, v: matracies and vector for constraint
%gradq(x): gradient of q as a function of x
%Z(x, c): solution to argmin_y h(x, y; c)
%options: 
%   eta: Lagrange penalty parameters, eta = 1 if none is given.
%   stepsize: output stepsize as a function of (x, y, c, s).  If not given,
%   optimal stepsize via numerical solver is used
%   maxIter: maximum number of iterations for outerloop
%   jmax: maximum number of iterations for innerloop
%   beta: beta computation routine.  Default is combination of FR and PR
%   values ('cases').  'PR' and 'FR' correspond to PR and FR beta
%   computation, respectively
%   xtrue: true value of unknown used to compute the relative error if
%   given

%Outputs:
%x: computed reconstruction
%info: algorithm iteration info
function [x, info] = vpalnl(q, r, gradq, Z, R, Q, v, options)

    info.version = '2025/08/26';

    if nargin < 1, fprintf('  Current vpal version: %s\n',info.version); return, end

    %initalize the default values
    maxIter = 1000; display = 'off'; tol = 1e-6; 

    if nargin == nargin(mfilename) % rewrite default parameters if needed
      for j = 1:2:length(options)
        eval([options{j},'= options{j+1};'])
      end
    end

    %if given an xtrue, we compute the relative error at each iteration
    if exist('xtrue', 'var')
        xtrue = xtrue;
        compute_error = true;
    else
        compute_error = false;
    end

    %stepsize for the line search.  either optimal, or give a function to
    %output it
    if exist('stepsize', 'var')
        stepsize_fcn = stepsize;
    else 
        stepsize = 'optimal';
    end

    %non-linear cg beta computation:
    if exist('beta', 'var')
        beta = beta;
    else
        beta = 'cases';
    end

    %eta is the Lagrange penalty parameter
    if exist('eta', 'var')
        eta = eta;
    else
        eta = 1;
    end

    %j max is how many iterations the inner loop runs for
    if exist('jmax', 'var')
        jmax = jmax;
    else
        jmax = 2;
        jmax;
    end

    %determine whether to keep information or not,  set to 0 and dispaly
    %'off' for time trials
    if nargout >= 2
        getInfo = 1; 
    else
        getInfo = 0; 
    end

    if getInfo 
      info.tol      = tol;
      info.maxIter  = maxIter;
    end

    % display and algorithm info
    if strcmp(display, 'iter') || strcmp(display,'final')
      fprintf('\nnon-linear vpal algorithm (c) Matthias Chung, Rosie Renaut, Jack Michael Solomon, May 2025\n');
      if strcmp(display,'final') == 0
        fprintf('\n %-5s %-6s %-14s \n','iter','loss','stop criteria');
      end
    end
    
    %get dimensions based on R, Q:
    n_x = size(Q, 2); n_y = size(R, 2); n_p = size(Q, 1);

    %initialize the Lagrange muiltipliers
    c = zeros(n_p, 1);

    %initalize x_0 if one is given:
    if exist('x', 'var')
        x = x;
    else
        x = zeros(n_x, 1);
    end

    %initalize y:
    if exist('y', 'var')
        y = y;
    else
        y = zeros(n_y, 1);
    end

    %initalize book-keeping variables
    f = inf; xOld = inf;       
    k = 1; eta2 = eta^2;

    %initalize gradient grad_x h(x, y; c):
    g = gradq(x) + eta2*(Q'*(Q*x + R*y - v + c));

    %main algorithm loop-------
    while 1

        %initalize s
        s = -g;

        %initalize iteration counter
        j = 0;

        while j < jmax

            %compute the stepsize
            if strcmp(stepsize, 'optimal')
                fcn = @(alpha) alpha_objective(alpha, x, c, s, Z, q, r, Q, R, v, eta2);
                if k == 1
                    alpha0 = 0.1;
                else 
                    alpha0 = alpha;
                end
                alpha = fminsearch(fcn, alpha0);
            else
                %may have to adjust per specific implementation
                alpha = stepsize_fcn(x, y, c, eta, s);
            end
            
            %update x
            x = x + alpha*s;

            %update y using Z(x)
            y = Z(x, c);
            
            %book-keeping
            gOld = g;

            %precompute Qx, Ry:
            Qx = Q*x;
            Ry = R*y;

            %compute new g (gradient of augmented lagrangian, h, with respect to x)
            g = gradq(x) + eta2*(Q'*(Qx + Ry - v + c));

            %compute beta and s:
            switch beta
                case 'FR'
                    beta = (g'*g)/(gOld'*gOld);
                case 'PR'
                    beta = (g'*(g - gOld))/(gOld'*gOld);
                case 'cases'
                    beta_FR = (g'*g)/(gOld'*gOld);
                    beta_PR = (g'*(g - gOld))/(gOld'*gOld);
                    if beta_PR < -beta_FR
                        beta = -beta_FR;
                    elseif abs(beta_PR) <= beta_FR
                        beta = beta_PR;
                    else
                        beta = beta_FR;
                    end
            end

            s = -g + beta*s;
            j = j+1;

        end

        %update the lagrange multipliers:
        c = c + (Qx + Ry - v);

        %book-keeping and info updates:
        fOld = f; f = q(x) + r(Qx); 

        stop1 = abs(fOld - f)  <= tol * (1 + f);
        stop2 = norm(xOld - x,'inf') <= sqrt(tol) * (1 + norm(x,'inf'));
        stop3 = k > maxIter-1;
        stop4 = f > 10e20; %check fcn is diverging

        if strcmp(display,'iter') % display iteration results
            fprintf('\n %5d %14.6e %4d%1d%1d%1d \n', k, f, stop1, stop2, stop3);
        end

        if (stop1 && stop2) || (stop1) || (stop3) || (stop4) % check stop criteria
            if stop3
              warning('Matlab:vpal:maxIter',...
                'Maximum number of iterations reached. Return with recent values.')
            end
            if stop4
                warning('Iteration diverging, Terminating')
            end
            break;
        end

        %can add additional information if desired
        if getInfo
            info.f(k)        = f;
            info.alpha(k)    = alpha;
            info.stop(:,k)   = [stop1;stop2;stop3];
            if compute_error
                info.rel_error(k) = norm(x - xtrue)/norm(xtrue);
            end
        end
        
        xOld = x; k = k+1;

    end
    if (strcmp(display,'iter') || strcmp(display,'final')) && ~stop3 % display
      fprintf('\nLocal minimizer found. Function value is %1.8e.\n', f);
    end

end


function value = alpha_objective(alpha, x, c, s, Z, q, r, Q, R, v, eta2)
    value = q(x + alpha*s) + r((Z(x + alpha*s, c))) + (eta2/2)*norm(Q*(x + alpha*s) + R*Z(x + alpha*s, c) - v + c, 2)^2;
end