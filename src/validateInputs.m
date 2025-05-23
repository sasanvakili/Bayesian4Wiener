function settings = validateInputs(model, settings, vecYbar)
% This function validates the input arguments (types, dimensions, etc.)
% according to the requirements specified in the paper:
%   "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".
%
% The validation ensures that the inputs conform to the expected format
% and constraints for the algorithm to function correctly.
%
% Paper: https://arxiv.org/abs/2504.05490
% Requirements: Bayesian4Wiener library (see README)
% ----------------------------------------------------------------------------------
% @author: Sasan Vakili
% @date: April 2025

if ~isstruct(model)
    error('First input argument must be a struct');
end

allowedFields = {'numState', 'numInput', 'numTheta', 'trajectoryT', ...
        'matrixAbar', 'matrixBbar', 'vecUbar', 'allVecFreq', 'muTheta', ... 
        'sigmaTheta', 'sigmaVbar', 'sigmaWbar'};
fields = fieldnames(model);
if ~(numel(fields) == numel(allowedFields) && ...
        all(ismember(fields, allowedFields)) && ...
        all(ismember(allowedFields, fields)))
    error(['model must have only the fields: numState, numInput, ' ...
        'numTheta, trajectoryT, matrixAbar, matrixBbar, vecUbar, allVecFreq,' ...
        ' muTheta, sigmaTheta, sigmaVbar, sigmaWbar']);
end

if ~(ismatrix(model.matrixAbar) && all(size(model.matrixAbar) == ...
        [model.numState*(model.trajectoryT+1), ...
        model.numState*(model.trajectoryT+1)]))
    error(['matrix Abar: dimension mismatch. ' ...
        'Expected size: [numState*(trajectoryT+1) by numState*(trajectoryT+1)].']);
end

if ~(ismatrix(model.matrixBbar) && all(size(model.matrixBbar) == ...
        [model.numState*(model.trajectoryT+1), ...
        model.numState+model.numInput*model.trajectoryT]))
    error(['matrix Bbar: dimension mismatch. ' ...
        'Expected size: [numState*(trajectoryT+1) by numState+numInput*trajectoryT].']);
end

if ~(isvector(model.vecUbar) && all(size(model.vecUbar) == ...
        [model.numState+model.numInput*model.trajectoryT 1]))
    error(['vector Ubar: dimension mismatch. ' ...
        'Expected size: [numState+numInput*trajectoryT by 1].']);
end

if ~(ismatrix(model.allVecFreq) && all(size(model.allVecFreq) == [model.numState model.numTheta-1]))
    error(['allVecFreq: dimension mismatch. ' ...
        'Expected size: [numState by numTheta-1].']);
end

zeroFreq = all(model.allVecFreq == 0, 1);
if any(zeroFreq)
    index = find(zeroFreq);
    error(['allVecFreq has at least one column of all zeros. ', ...
    'Zero frequencies are not expected. All-zero columns: %s'], mat2str(index));
end

if ~(isvector(model.muTheta) && all(size(model.muTheta) == [model.numTheta 1]))
    error(['vector muTheta: dimension mismatch. ' ...
        'Expected size: [numTheta by 1].']);
end

if ~(ismatrix(model.sigmaTheta) && all(size(model.sigmaTheta) == ...
        [model.numTheta model.numTheta]))
    error(['matrix sigmaTheta: dimension mismatch. ' ...
        'Expected size: [numTheta by numTheta].']);
end

if ~(ismatrix(model.sigmaVbar) && all(size(model.sigmaVbar) == ...
        [model.trajectoryT+1 model.trajectoryT+1]))
    error(['matrix sigmaVbar: dimension mismatch. ' ...
        'Expected size: [trajectoryT+1 by trajectoryT+1].']);
end

if ~(ismatrix(model.sigmaWbar) && all(size(model.sigmaWbar) == ...
        [model.numState*(model.trajectoryT+1), ...
        model.numState*(model.trajectoryT+1)]))
    error(['matrix sigmaWbar: dimension mismatch. ' ...
        'Expected size: [numState*(trajectoryT+1) by numState*(trajectoryT+1)].']);
end

if ~isstruct(settings)
    error('Second input argument must be a struct');
end

allowedFields = {'mode', 'activeLearning', 'verbose'};
fields = fieldnames(settings);
if ~(numel(fields) == numel(allowedFields) && ...
        all(ismember(fields, allowedFields)) && ...
        all(ismember(allowedFields, fields)))
    error('settings: must have only the fields: mode, activeLearning, verbose');
end

if ~ismember(settings.verbose, [0, 1, 2])
    error('Invalid verbose value: must be 0, 1, or 2');
end

validModes = {'estimatorOnly' 'estimateTheta'  'activeLearning'};
settings.mode = validatestring(settings.mode, validModes);

if strcmp(settings.mode, 'estimateTheta')
    if ~(isvector(vecYbar) && all(size(vecYbar) ...
            == [model.trajectoryT+1, 1]))
        error(['vecYbar: dimension mismatch. ' ...
            'Expected size: [trajectoryT+1 by 1].']);
    end
end

if strcmp(settings.mode, 'activeLearning')
    allowedFields = {'gradTol', 'costTol', 'maxIter', 'alpha', 'beta', ...
        'applyToInitX', 'existConstraint', 'vecUmax', 'vecUmin', ...
        'maxInitState', 'minInitState'};
    fields = fieldnames(settings.activeLearning);
    if ~(numel(fields) == numel(allowedFields) && ...
            all(ismember(fields, allowedFields)) && ...
            all(ismember(allowedFields, fields)))
        error(['settings.activeLearning: must have only the fields: ' ...
            'gradTol, costTol, maxIter, alpha, beta, applyToInitX, ' ...
            'existConstraint, vecUmax, vecUmin, maxInitState, minInitState']);
    end

    if ~(isnumeric(settings.activeLearning.gradTol) && ...
            isscalar(settings.activeLearning.gradTol))
        error(['Invalid gradient tolerance: settings.activeLearning.gradTol ' ...
            'must be a scalar numerical value (e.g., 1e-6).'])
    end

    if ~(isnumeric(settings.activeLearning.costTol) && ...
            isscalar(settings.activeLearning.costTol))
        error(['Invalid cost tolerance: settings.activeLearning.costTol ' ...
            'must be a scalar numerical value (e.g., 1e-6).'])
    end

    if ~(isnumeric(settings.activeLearning.maxIter) && ...
            isscalar(settings.activeLearning.maxIter))
        error(['Invalid maximum iteration tolerance: ' ...
            'settings.activeLearning.maxIter must be a scalar numerical ' ...
            'value (e.g., 10000).'])
    end

    if ~(isnumeric(settings.activeLearning.alpha) && ...
            isscalar(settings.activeLearning.alpha))
        error(['Invalid alpha initialization: ' ...
            'settings.activeLearning.alpha must be a scalar numerical ' ...
            'value (e.g., 1e-10).'])
    end

    if ~(isnumeric(settings.activeLearning.beta) && ...
            isscalar(settings.activeLearning.beta))
        error(['Invalid beta initialization: ' ...
            'settings.activeLearning.beta must be a scalar numerical ' ...
            'value (e.g., 1e100).'])
    end

    if ~(islogical(settings.activeLearning.existConstraint) || ...
            (isnumeric(settings.activeLearning.existConstraint) && ...
            ismember(settings.activeLearning.existConstraint, [0 1])))
        error('settings.activeLearning.existConstraint must be logical or 0/1.');
    end

    if (isnumeric(settings.activeLearning.existConstraint) && ...
            ismember(settings.activeLearning.existConstraint, [0 1]))
        settings.activeLearning.existConstraint = logical( ...
            settings.activeLearning.existConstraint);
    end

    if ~(islogical(settings.activeLearning.applyToInitX) || ...
            (isnumeric(settings.activeLearning.applyToInitX) && ...
            ismember(settings.activeLearning.applyToInitX, [0 1])))
        error('settings.activeLearning.applyToInitX must be logical or 0/1.');
    end

    if (isnumeric(settings.activeLearning.applyToInitX) && ...
            ismember(settings.activeLearning.applyToInitX, [0 1]))
        settings.activeLearning.applyToInitX = logical( ...
            settings.activeLearning.applyToInitX);
    end
    
    if (settings.activeLearning.existConstraint)
        if ~(isvector(settings.activeLearning.vecUmax) && ...
                all(size(settings.activeLearning.vecUmax) == [model.numInput 1]))
            error(['settings.activeLearning.vecUmax: dimension mismatch. ' ...
                'Expected size: [numInput by 1].']);
        end
        if ~(isvector(settings.activeLearning.vecUmin) && ...
                all(size(settings.activeLearning.vecUmin) == [model.numInput 1]))
            error(['settings.activeLearning.vecUmin: dimension mismatch. ' ...
                'Expected size: [numInput by 1].']);
        end
        if (settings.activeLearning.applyToInitX)
            if ~(isvector(settings.activeLearning.maxInitState) && ...
                    all(size(settings.activeLearning.maxInitState) == [model.numState 1]))
                error(['settings.activeLearning.maxInitState: dimension mismatch. ' ...
                    'Expected size: [numState by 1].']);
            end
            if ~(isvector(settings.activeLearning.minInitState) && ...
                    all(size(settings.activeLearning.minInitState) == [model.numState 1]))
                error(['settings.activeLearning.minInitState: dimension mismatch. ' ...
                    'Expected size: [numState by 1].']);
            end
        end
    end
end
end