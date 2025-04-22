function [sigmaPhi, muPhi] = fourierDBS(numTheta, allVecFreq, matrixAi, ...
    vecBbarUbar, sigmaWbar, matrixAj)
% This function computes the Fourier "Dynamic basis statistics" (DBS) explicit 
% expression defined in Lemma 4.1 of Section 4 from the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% This computation is critical for implementing the proposed optimal Bayesian estimator.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README for details)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

sigmaPhi = zeros(numTheta, numTheta);

allFreqAi = (allVecFreq')*matrixAi;
allFreqAj = (allVecFreq')*matrixAj;
sumFreqAij = permute(allFreqAi, [1 3 2]) + permute(allFreqAj, [3 1 2]);
sumFreqAij = reshape(sumFreqAij, (numTheta-1)*(numTheta-1), 1, size(matrixAi, 2));
sumFreqAij = permute(sumFreqAij, [2 3 1]);
sumFrqAijU = pagemtimes(sumFreqAij, vecBbarUbar);
sumFrqAijU = reshape(sumFrqAijU, (numTheta-1), (numTheta-1));

subFreqAij = permute(allFreqAi, [1 3 2]) - permute(allFreqAj, [3 1 2]);
subFreqAij = reshape(subFreqAij, (numTheta-1)*(numTheta-1), 1, size(matrixAi, 2));
subFreqAij = permute(subFreqAij, [2 3 1]);
subFrqAijU = pagemtimes(subFreqAij, vecBbarUbar);
subFrqAijU = reshape(subFrqAijU, (numTheta-1), (numTheta-1));

sumFrqAijW = pagemtimes(sumFreqAij, sigmaWbar);
sumFrqAijWf = pagemtimes(sumFrqAijW, 'none', sumFreqAij, 'transpose');
sumFrqAijWf = reshape(sumFrqAijWf, (numTheta-1), (numTheta-1));

subFrqAijW = pagemtimes(subFreqAij, sigmaWbar);
subFrqAijWf = pagemtimes(subFrqAijW, 'none', subFreqAij, 'transpose');
subFrqAijWf = reshape(subFrqAijWf, (numTheta-1), (numTheta-1));

charstcProd = exp(-(1/2)*sum(((allVecFreq')*matrixAi*sigmaWbar).* ...
    ((allVecFreq')*matrixAi), 2))*...
    (exp(-(1/2)*sum(((allVecFreq')*matrixAj*sigmaWbar).* ...
    ((allVecFreq')*matrixAj), 2))');

sigmaPhi(2:end, 2:end) = (exp(1i*sumFrqAijU)+exp(-1i*sumFrqAijU)).*( ...
    exp(-(1/2)*sumFrqAijWf) - charstcProd) + ...
    (exp(1i*subFrqAijU)+exp(-1i*subFrqAijU)).*( ...
    exp(-(1/2)*subFrqAijWf) - charstcProd);

if any(abs(imag(sigmaPhi)) >= 1e-6)
    error(['sigmaPhi has non-negligible imaginary parts ' ...
        '(max abs(imag): %.2e)'], max(abs(imag(sigmaPhi))));
end
sigmaPhi = real(sigmaPhi);

if (nargout > 1)
    muPhi = zeros(numTheta, 1);
    muPhi(1) = 1;
    muPhi(2:end) = (exp(1i*(allVecFreq')*matrixAi*vecBbarUbar)+ ...
        exp(-1i*(allVecFreq')*matrixAi*vecBbarUbar)).*...
        exp(-(1/2)*sum(((allVecFreq')*matrixAi*sigmaWbar) .* ...
        ((allVecFreq')*matrixAi), 2));
    if any(abs(imag(muPhi)) >= 1e-6)
        error(['muPhi has non-negligible imaginary parts ' ...
            '(max abs(imag): %.2e)'], max(abs(imag(muPhi))));
    end
    muPhi = real(muPhi);
end

end