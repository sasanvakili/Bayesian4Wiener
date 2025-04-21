% This example replicates the experimental setup from the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model"
% The following code demonstrates the use of the Bayesian4Wiener library to compute:
%   1) The optimal Bayesian MMSE affine estimator (BMS) from Section 3.2;
%   2) Its integration with Bayesian active learning (BAL) from Section 5;
% as illustrated in Benchmarks 2 of the paper.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

% Add library path
addpath('../src');
% Load experiment data:
load("./experimentData.mat")
% Process noise in the experiment data includes:
%   1. vecWbar2: random values generated with sigmaWbar2 (sigmaW = 0.001)
%   2. vecWbar3: random values generated with sigmaWbar3 (sigmaW = 0.01)
% The following code demonstrates BMS and BAL for the first realization of 
% vecVbar, vecWbar2 and theta.
model.sigmaWbar = model.sigmaWbar2;
model = rmfield(model, 'sigmaWbar2');
model = rmfield(model, 'sigmaWbar3');

%% BMS:
% Case 1 - Estimator Only:
disp(['Case 1 - Estimator Only: execute the Bayesian4Wiener library with' ...
    ' the model struct (no measurements)'])
% The 'estimatorOnly' mode computes the BMS estimator parameters using only
% the input sequence (BMS estimator).
settings = struct;
settings.mode = 'estimatorOnly';
settings.verbose = 0;
settings.activeLearning = [];
vecYbar = [];
[estimator1, optimizer1, optimalUbar1, thetaEstimate1] = Bayesian4Wiener(model, ...
    settings, vecYbar)
disp('-------------------------------------------------------------------------------')

% Case 2 - Estimate Theta:
disp(['Case 2 - Estimate Theta: execute the Bayesian4Wiener library with ' ...
    'the model struct and measurements'])
% Generate measurements from random variable data and the input sequence.
% Create system states:
vecXbar = model.matrixAbar*(model.matrixBbar*model.vecUbar + vecWbar2(:,1));
% Create measurements:
vecYbar = fourierObservation(model.allVecFreq, trueTheta(:,1), vecXbar, ...
    model.numState)+vecVbar(:,1);
% The 'estimateTheta' mode computes unknown theta estimates using the input
% sequence and corresponding measurements (BMS estimate).
settings = struct;
settings.mode = 'estimateTheta';
settings.verbose = 1;
settings.activeLearning = [];
[estimator2, ~, ~, thetaEstimate2] = Bayesian4Wiener(model, ...
    settings, vecYbar);
estimationErrBMS = sum((trueTheta(:,1)-thetaEstimate2).^2)
disp('-------------------------------------------------------------------------------')

%% BAL:
% Case 3 - Active Learning:
disp(['Case 3 - Optimal Input: execute the Bayesian4Wiener library to ' ...
    'perform active learning using the model struct' ...
    ' and the initial inputs']);
% The 'activeLearning' mode designs optimal input signals (BAL estimator)
% using the given input sequence as initialization.
settings = struct;
settings.mode = 'activeLearning';
settings.verbose = 2;
settings.activeLearning.gradTol = 1e-6;
settings.activeLearning.costTol = 1e-6;
% To reproduce results approximately matching the paper's findings, use:
%   settings.activeLearning.maxIter = 10000;
% The current configuration demonstrates error reduction with minimal iterations.
settings.activeLearning.maxIter = 5; 
settings.activeLearning.alpha = 1e-10;
settings.activeLearning.beta = 1e100;
settings.activeLearning.applyToInitX = false;
settings.activeLearning.existConstraint = true;
settings.activeLearning.vecUmax = inputConstraint.vecUmax;
settings.activeLearning.vecUmin = inputConstraint.vecUmin;
settings.activeLearning.maxInitState = [];
settings.activeLearning.minInitState = [];
vecYbar = [];
[estimator3, optimizer3, optimalUbar3, ~] = Bayesian4Wiener(model, ...
    settings, vecYbar);
disp('-------------------------------------------------------------------------------')

% Case 4 - Estimate Theta:
disp(['Case 4 - Estimate theta with optimal input: ' ...
    'execute the Bayesian4Wiener library using the model struct and ' ...
    'measurements from optimal inputs']);
% Generate measurements from random variable data and the optimal input.
model.vecUbar = optimalUbar3;
% Create system states:
vecXbar = model.matrixAbar*(model.matrixBbar*model.vecUbar + vecWbar2(:,1));
% Create measurements:
vecYbar = fourierObservation(model.allVecFreq, trueTheta(:,1), vecXbar, ...
    model.numState)+vecVbar(:,1);
% The 'estimateTheta' mode computes estimates of unknown theta using the
% optimal input sequence and their corresponding measurements (BAL estimates).
settings = struct;
settings.mode = 'estimateTheta';
settings.verbose = 1;
settings.activeLearning = [];
[estimator4, ~, ~, thetaEstimate4] = Bayesian4Wiener(model, ...
    settings, vecYbar);
estimationErrBAL = sum((trueTheta(:,1)-thetaEstimate4).^2)
disp('-------------------------------------------------------------------------------')

% -------------------------------------------------------------------------------
% Fourier observation model:
function vecY = fourierObservation(allVecFreq, theta, vecX, numState)
dataLen = length(vecX)/numState;
vecY = zeros(dataLen,1);
theta0 = repmat(theta(1),dataLen,1);
theta(1) = [];
parfor i=0:dataLen-1
    tempX = vecX(numState*i+1:numState*i+numState);
    vecY(i+1) = (theta')*(exp(1i*(allVecFreq')*tempX)+exp(-1i*(allVecFreq')*tempX));
end
vecY = vecY+theta0;
if ~isempty(vecY(imag(vecY)>=1e-12))
    error('Error: vectorY has imaginary part!')
end
vecY = real(vecY);
end





