function [estimator, optimizer, optimalUbar, thetaEstimate] = Bayesian4Wiener(model, settings, vecYbar)
% This library is implemented according to the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model"
%
% It computes either of the following:
%   1) Optimal Bayesian MMSE affine estimator parameters;
%   2) Optimal Bayesian MMSE affine estimator parameters and theta estimates;
%   3) Optimal input using the active learning algorithm and its
%   corresponding optimal Bayesian MMSE affine estimator parameters.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

settings = validateInputs(model, settings, vecYbar);

estimator = struct;
switch settings.mode
    case 'estimatorOnly'
        if (settings.verbose >= 1)
            msg = 'Computing only estimator parameters:';
            disp(msg)
        end
        [estimator] = bayesAffineEstimator(model);
        optimizer = [];
        optimalUbar = [];
        thetaEstimate = [];
    case 'estimateTheta'
        if (settings.verbose >= 1)
            msg = 'Computing estimator parameters and theta estimates:';
            disp(msg)
        end
        [estimator] = bayesAffineEstimator(model);
        thetaEstimate = estimator.matrixPsi*vecYbar+estimator.vecPsi;
        optimizer = [];
        optimalUbar = [];
    case 'activeLearning'
        if (settings.verbose >= 1)
            msg = 'Computing optimal input and its corresponding estimator parameters:';
            disp(msg)
        end
        [estimator, optimizer, optimalUbar] = adaptiveGradientDescent(model, settings);
        thetaEstimate = [];
end
end