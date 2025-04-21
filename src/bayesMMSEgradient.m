function gradientUbar = bayesMMSEgradient(model, estimator)
% This function computes the gradient of the Bayesian MMSE estimation error
% as defined in Section 5 of the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% This gradient computation enables the adaptive gradient descent algorithm
% to find alocally optimal input signal that minimizes the Bayesian MMSE
% estimation error.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

costPhibar = 2*((estimator.matrixPsi).')*( ...
    (estimator.matrixPsi)*((estimator.matrixPhibar).')-eye(model.numTheta) ...
    )*(model.sigmaTheta);
costMatrixM = ((estimator.matrixPsi).')*(estimator.matrixPsi);
matrixMsigmaPhi = (model.sigmaTheta+((model.muTheta)*((model.muTheta).')));

phibarVecU = zeros(model.numTheta, model.trajectoryT+1, ...
    model.numState+(model.numInput*model.trajectoryT));
upMatrixMvecU = zeros(model.trajectoryT+1, model.trajectoryT+1, ...
    model.numState+(model.numInput*model.trajectoryT));
lowMatrixMvecU = zeros(model.trajectoryT+1, model.trajectoryT+1, ...
    model.numState+(model.numInput*model.trajectoryT));
parfor i=0:model.trajectoryT
    matrixAi = model.matrixAbar(model.numState*i+1:model.numState*i+model.numState, :);
    rowsMvecU = zeros(1, model.trajectoryT+1, ...
        model.numState+(model.numInput*model.trajectoryT));
    colsMvecU = zeros(model.trajectoryT+1, 1, ...
        model.numState+(model.numInput*model.trajectoryT));
    tempMvecU = zeros(1, model.numState+(model.numInput*model.trajectoryT));
    for j=i:model.trajectoryT
        matrixAj = model.matrixAbar(model.numState*j+1:model.numState*j+model.numState, :);
        if (j == i)
            [sigmaPhiVecUbar, muPhiVecUbar] = fourierDBSgradient(model.numTheta, ...
                model.allVecFreq, matrixAi, model.matrixBbar, model.vecUbar, ...
                model.sigmaWbar, matrixAj);
            phibarVecU(:, i+1, :) = muPhiVecUbar;
        else
            sigmaPhiVecUbar = fourierDBSgradient(model.numTheta, ...
                model.allVecFreq, matrixAi, model.matrixBbar, model.vecUbar, ...
                model.sigmaWbar, matrixAj);
        end
        tempMsigmaPhiU = pagemtimes(matrixMsigmaPhi,sigmaPhiVecUbar);
        for l=1:size(tempMsigmaPhiU,3)
            tempMvecU(l) = trace(tempMsigmaPhiU(:, :, l));
        end
        rowsMvecU(1, j+1, :) = tempMvecU;
        if (j ~= i)
            colsMvecU(j+1, 1, :) = tempMvecU;
        end
    end
    upMatrixMvecU(i+1, :, :) = rowsMvecU;
    lowMatrixMvecU(:, i+1, :) = colsMvecU;
end
matrixMvecU = upMatrixMvecU+lowMatrixMvecU;

tempPhiUbar = pagemtimes(costPhibar,phibarVecU);
tempMvecUbar = pagemtimes(costMatrixM,matrixMvecU);
costPhiUbar = zeros(model.numState+(model.numInput*model.trajectoryT), 1);
costMvecUbar = zeros(model.numState+(model.numInput*model.trajectoryT), 1);
parfor t=1:size(tempPhiUbar,3)
    costPhiUbar(t) = trace(tempPhiUbar(:,:,t));
    costMvecUbar(t) = trace(tempMvecUbar(:,:,t));
end
gradientUbar = costPhiUbar+costMvecUbar;
end
