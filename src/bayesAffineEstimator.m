function estimator = bayesAffineEstimator(model)
% This function computes the Optimal Bayesian MMSE affine estimator defined in Theorem
% 3.2 of Section 3 from the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% It returns the estimator parameters and optimal cost as derived in the theorem.
% 
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

estimator = struct;
vecBbarUbar = model.matrixBbar*model.vecUbar;

matrixPhibar = zeros(model.numTheta, model.trajectoryT+1);
matrixM = zeros(model.trajectoryT+1, model.trajectoryT+1);
parfor i=0:model.trajectoryT
    matrixAi = model.matrixAbar(model.numState*i+1:model.numState*i+model.numState, :);
    rowsMatrixM = zeros(1, model.trajectoryT+1);
    for j=i:model.trajectoryT
        matrixAj = model.matrixAbar(model.numState*j+1:model.numState*j+model.numState, :);
        if (j == i)
            [sigmaPhi, muPhi] = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
            matrixPhibar(:, i+1) = muPhi;
        else
            sigmaPhi = fourierDBS(model.numTheta, model.allVecFreq, matrixAi, ...
                vecBbarUbar, model.sigmaWbar, matrixAj);
        end
        rowsMatrixM(:, j+1) = trace(sigmaPhi*(model.sigmaTheta+(model.muTheta)*(model.muTheta')));
    end
    matrixM(i+1, :) = rowsMatrixM;
end
matrixM = matrixM+triu(matrixM,1).';

matrixPsi = (model.sigmaTheta*matrixPhibar)/(((matrixPhibar.')*...
    model.sigmaTheta*matrixPhibar)+matrixM+model.sigmaVbar);
vecPsi = model.muTheta - matrixPsi*(matrixPhibar.')*model.muTheta;
optimalErr = trace(model.sigmaTheta - (matrixPsi*(matrixPhibar.')*model.sigmaTheta));

estimator.matrixPhibar = matrixPhibar;
estimator.matrixM = matrixM;
estimator.matrixPsi = matrixPsi;
estimator.vecPsi = vecPsi;
estimator.optimalErr = optimalErr;
end