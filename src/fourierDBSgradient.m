function [sigmaPhiVecUbar, muPhiVecUbar] = fourierDBSgradient(numTheta, ...
    allVecFreq, matrixAi, matrixBbar, vecUbar, sigmaWbar, matrixAj)
% This function computes the Fourier "Dynamic basis statistics" (DBS)
% gradient as defined in Section 5 of the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% This computation is critical for implementing the proposed active learning algorithm.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

sigmaPhiVecUbar = zeros(numTheta, numTheta, length(vecUbar));

allFreqAi = (allVecFreq')*matrixAi;
allFreqAj = (allVecFreq')*matrixAj;
sumFreqAij = permute(allFreqAi, [1 3 2]) + permute(allFreqAj, [3 1 2]);
sumFreqAij = reshape(sumFreqAij, (numTheta-1)*(numTheta-1), 1, size(matrixAi, 2));
sumFreqAij = permute(sumFreqAij, [2 3 1]);
subFreqAij = permute(allFreqAi, [1 3 2]) - permute(allFreqAj, [3 1 2]);
subFreqAij = reshape(subFreqAij, (numTheta-1)*(numTheta-1), 1, size(matrixAi, 2));
subFreqAij = permute(subFreqAij, [2 3 1]);

sumFrqAijU = pagemtimes(sumFreqAij, matrixBbar*vecUbar);
sumFrqAijU = reshape(sumFrqAijU, (numTheta-1), (numTheta-1));
subFrqAijU = pagemtimes(subFreqAij, matrixBbar*vecUbar);
subFrqAijU = reshape(subFrqAijU, (numTheta-1), (numTheta-1));

sumFrqAijW = pagemtimes(sumFreqAij, sigmaWbar);
sumFrqAijWf = pagemtimes(sumFrqAijW, 'none', sumFreqAij, 'transpose');
sumFrqAijWf = reshape(sumFrqAijWf, (numTheta-1), (numTheta-1));
subFrqAijW = pagemtimes(subFreqAij, sigmaWbar);
subFrqAijWf = pagemtimes(subFrqAijW, 'none', subFreqAij, 'transpose');
subFrqAijWf = reshape(subFrqAijWf, (numTheta-1), (numTheta-1));

sumFrqAijB = pagemtimes(sumFreqAij, matrixBbar);
sumFrqAijB = permute(sumFrqAijB, [3 1 2]);
sumFrqAijB = reshape(sumFrqAijB, (numTheta-1), (numTheta-1), length(vecUbar));
subFrqAijB = pagemtimes(subFreqAij, matrixBbar);
subFrqAijB = permute(subFrqAijB, [3 1 2]);
subFrqAijB = reshape(subFrqAijB, (numTheta-1), (numTheta-1), length(vecUbar));

charstcProd = exp(-(1/2)*sum(((allVecFreq')*matrixAi*sigmaWbar).* ...
    ((allVecFreq')*matrixAi), 2))*...
    (exp(-(1/2)*sum(((allVecFreq')*matrixAj*sigmaWbar).* ...
    ((allVecFreq')*matrixAj), 2))');

sigmaPhiVecUbar(2:end, 2:end, :) = (1i*exp(1i*sumFrqAijU)-1i*exp(-1i*sumFrqAijU)).*( ...
    exp(-(1/2)*sumFrqAijWf) - charstcProd).*(sumFrqAijB) + ...
    (1i*exp(1i*subFrqAijU)-1i*exp(-1i*subFrqAijU)).*( ...
    exp(-(1/2)*subFrqAijWf) - charstcProd).*(subFrqAijB);

if any(abs(imag(sigmaPhiVecUbar)) >= 1e-6)
    error(['sigmaPhi has non-negligible imaginary parts ' ...
        '(max abs(imag): %.2e)'], max(abs(imag(sigmaPhiVecUbar))));
end
sigmaPhiVecUbar = real(sigmaPhiVecUbar);

if (nargout > 1)
    muPhiVecUbar = zeros(numTheta, 1, length(vecUbar));
    muPhiVecUbar(2:end, 1, :) = (1i*exp(1i*(allVecFreq')*matrixAi* ...
        matrixBbar*vecUbar)-1i*exp(-1i*(allVecFreq')*matrixAi* ...
        matrixBbar*vecUbar)).*exp(-(1/2)*sum(((allVecFreq')*matrixAi ...
        *sigmaWbar).*((allVecFreq')*matrixAi), 2)).*( ...
        (allVecFreq')*matrixAi*matrixBbar);
    if any(abs(imag(muPhiVecUbar)) >= 1e-6)
        error(['muPhi has non-negligible imaginary parts ' ...
            '(max abs(imag): %.2e)'], max(abs(imag(muPhiVecUbar))));
    end
    muPhiVecUbar = real(muPhiVecUbar);
end
end