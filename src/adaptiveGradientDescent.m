function [estimator, optimizer, optimalUbar] = adaptiveGradientDescent(model, settings)
% This function the adaptive gradient descent algorithm for active learning
% as defined in Section 5 of the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% This algorithm optimizes input signals to minimize the Bayesian MMSE
% estimation error for Wiener system identification. It returns estimator
% parameters, Optimization metrics (e.g., convergence status), Optimal
% input signal.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

% Decision variable: vecUbar
if (settings.verbose >= 2)
    msg = 'Adaptive stepsize Projected Gradient Descent:';
    disp(msg)
end
% Linear input constraint
if (settings.activeLearning.existConstraint)
    if ~(settings.activeLearning.applyToInitX)
        %InitState is not optimized!
        maxConstraint = [model.vecUbar(1:model.numState); 
            repmat(settings.activeLearning.vecUmax, model.trajectoryT, 1)];
        minConstraint = [model.vecUbar(1:model.numState);
            repmat(settings.activeLearning.vecUmin, model.trajectoryT, 1)];
    else
        maxConstraint = [settings.activeLearning.maxInitState;
            repmat(settings.activeLearning.vecUmax, model.trajectoryT, 1)];
        minConstraint = [settings.activeLearning.minInitState;
            repmat(settings.activeLearning.vecUmin, model.trajectoryT, 1)];
    end
end
vecCost = zeros(1, 2);
numIter = 0;
estimator = bayesAffineEstimator(model);
vecCost(2) = estimator.optimalErr;
while true
    numIter = numIter+1;
    gradientUbar = bayesMMSEgradient(model, estimator);
    if ~(settings.activeLearning.applyToInitX)
        %InitState is not optimized!
        gradientUbar(1:model.numState) = zeros(model.numState, 1);
    end
    if (numIter < 2)
        beta = settings.activeLearning.beta;
        alpha = settings.activeLearning.alpha;
    else
        % alpha Update:
        term1 = sqrt(1+beta)*alpha;
        term2 = (norm(model.vecUbar - prevUbar))/...
            (2*(norm(gradientUbar-prevGradUbar)));
        prevAlpha = alpha;
        alpha = min(term1,term2);
        beta = alpha/prevAlpha;
    end
    prevUbar = model.vecUbar;
    % Descent Update
    model.vecUbar = model.vecUbar - alpha*gradientUbar;
    if (settings.activeLearning.existConstraint)
        % Projection to the linear input constraint
        model.vecUbar(model.vecUbar > maxConstraint) = ...
            maxConstraint(model.vecUbar > maxConstraint);
        model.vecUbar(model.vecUbar < minConstraint) = ...
            minConstraint(model.vecUbar < minConstraint);
        if (settings.verbose >= 2)
            if ~(isempty(model.vecUbar(model.vecUbar > maxConstraint)) ...
                    && isempty(model.vecUbar(model.vecUbar < minConstraint)))
                msg = ['Iteration - ', num2str(numIter), ': ', ...
                    'vecUbar is projected onto its constraints.'];
                disp(msg);
            end
        end
    end
    estimator = bayesAffineEstimator(model);
    vecCost(1) = vecCost(2);
    vecCost(2) = estimator.optimalErr;
    prevGradUbar = gradientUbar;
    if (settings.verbose >= 2)
        msg = ['Iteration - ', num2str(numIter), ': ', ...
          'Gradient norm: ', num2str(norm(gradientUbar)), ', ', ...
          'Cost: ', num2str(vecCost(2))];
        disp(msg)
    end
    costDiff = abs(vecCost(2) - vecCost(1));
    if ((norm(gradientUbar) < settings.activeLearning.gradTol) ...
            && (costDiff < settings.activeLearning.costTol))
        status = 'converged';
        if (settings.verbose >= 2)
            msg = ['Converged: Gradient and cost tolerances met.', newline, ...
                'Total iterations: ', num2str(numIter), newline, ...
                'Final gradient norm: ', num2str(norm(gradientUbar)), ...
                newline, 'Final cost: ', num2str(vecCost(2))];
            disp(msg)
        end
        break
    end
    if (numIter >= settings.activeLearning.maxIter)
        status = 'max iterations';
        if (settings.verbose >= 2)
            msg = ['Stopped at maximum iteration.', newline, ...
                'Iterations: ', num2str(numIter), newline, ...
                'Final gradient norm: ', num2str(norm(gradientUbar)), ...
                newline, 'Final cost: ', num2str(vecCost(2))];
            disp(msg)
        end
        break
    end
end
optimizer.alpha = alpha;
optimizer.beta = beta;
optimizer.status = status;
optimizer.totalIter = numIter;
optimizer.gradient = gradientUbar;
optimizer.optimalCost = vecCost(2);
optimalUbar = model.vecUbar;
end
