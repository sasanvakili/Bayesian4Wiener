# Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model
Consider a known discrete-time linear time-varying dynamical system where the states at time $t$ are observed through an _unknown_ observation model:

$$
\mathrm{x} _{t+1} = \mathrm{A} _{t} \mathrm{x} _{t} + \mathrm{B} _{t} \mathrm{u} _{t} + \mathrm{w} _{t+1},
$$
$$
\mathrm{y} _{t} = \sum\limits _{n=0}^{N} \theta _{n} \phi _{n}( \mathrm{x} _{t}) + \mathrm{v} _{t} = \langle {\phi(\mathrm{x} _{t}), \theta} \rangle,
$$

where $t = \\{ 0, \\ldots, T \\}$, $\mathrm{x} _{t} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the vector of state variables, $\mathrm{A} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{x}} }$ is the state transition matrix, $\mathrm{u} _{t} \in \mathbb{R} ^{n _{\mathrm{u}}}$ is the vector of inputs, $\mathrm{B} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{u}}}$ is the input matrix, and $\mathrm{w} _{t+1} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the process noise. Observations are made through the scalar output measurements $\mathrm{y} _{t} \in \mathbb{R}$, while $\mathrm{v} _{t} \in \mathbb{R}$ represents the measurement noise. The _known_ basis functions $\phi _{n}( \mathrm{x} _{t})$ is the following Fourier basis function:

$$ 
\begin{cases}
\phi_ {0} ( \mathrm{x} _{t} ) = 1 & n = 0 \\
\phi _{n}( \mathrm{x} _{t} ) = \sum\limits _{\ell \in \\{ -1, 1 \\}} \mathrm{exp} ( j \langle { \ell f _{n} , \mathrm{x} _{t} } \rangle ) & n \geq 1,
\end{cases}
$$

with _known_ frequencies $f _{n} \in \mathbb{R} ^{n _{\mathrm{x}}}$, and $n$ denotes the frequency index, while $\phi(\mathrm{x} _{t}) = \[ \phi _{0}(\mathrm{x} _{t}), \ldots, \phi _{N}(\mathrm{x} _{t}) \] ^{\mathsf{T}}$ is the vector of basis functions evaluated at $\mathrm{x} _{t}$.

- The **optimal Bayesian affine estimator** is of the form 

$$
\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) = \Psi^{\star} \overline{\mathrm{y}} + \psi^{\star},
$$

which estimates the model parameters $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ from the measurement data $\mathrm{y} _{t}$ at all time steps, represented as $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$. To compute $\Psi^{\star}$ and $\psi^{\star}$, the models are represented in _lifted matrix_ form. As such, the output function is

$$
\overline{\mathrm{y}} = \Phi ^{\mathsf{T}} \theta + \overline{\mathrm{v}},
$$

where $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$, $\Phi = \[ \phi(\mathrm{x} _{0}), \ldots, \phi(\mathrm{x} _{T}) \]$ is the basis aggregation matrix, $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$, and $\overline{\mathrm{v}} = \[ \mathrm{v} _{0}, \ldots, \\mathrm{v} _{T} \] ^{\mathsf{T}}$ is the measurement noise vector. The measurement noise follows $\overline{\mathrm{v}} \sim \mathbb{P} ( 0, \Sigma _{\mathrm{v}})$, where $\Sigma _{\mathrm{v}} = \mathrm{diag}( \Sigma _{\mathrm{v} _{0}}^{2}, \ldots, \Sigma _{\mathrm{v} _{T}}^{2})$. In addition, the prior information about $\theta$ is characterized by a probability distribution defined by its mean and covariance, $\mathbb{P} ( \mu _{\theta}, \Sigma _{\theta})$. In addition, the process model is represented as

$$
\overline{\mathrm{x}} = \overline{\mathrm{A}} (\overline{\mathrm{B}} \overline{\mathrm{u}} + \overline{\mathrm{w}}),
$$

where $\overline{\mathrm{x}} = \[ \mathrm{x} _{0} ^{\mathsf{T}}, \ldots, \mathrm{x} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ consists of the system states vector, $\overline{\mathrm{u}} = \[ \mu _{\mathrm{x} _{0}} ^{\mathsf{T}}, \mathrm{u} _{0} ^{\mathsf{T}}, \ldots, \mathrm{u} _{T-1} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the input vector with the mean of the initial state $\mu _{\mathrm{x} _{0}}$ as its first element, and $\overline{\mathrm{w}} = \[ \mathrm{w} _{0} ^{\mathsf{T}}, \mathrm{w} _{1} ^{\mathsf{T}}, \ldots, \mathrm{w} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ denotes the noise vector in which the first element corresponds to the uncertainty of the initial state. As such, $\mathrm{w} _{0} \sim \mathbb{P} (0, \Sigma _{\mathrm{x} _{0}} )$ and $\overline{\mathrm{w}}$ is a __zero-mean__ uncertainty, i.e., $\overline{\mathrm{w}} \sim \mathbb{P} ( 0 , \Sigma _{\overline{\mathrm{w}}} )$. The covariance matrix is diagonal only if $\mathrm{w} _{t}$ are independent, i.e., $\Sigma _{\overline{\mathrm{w}}} = \mathrm{diag} ( \Sigma _{\mathrm{x} _{0}}, \Sigma _{\mathrm{w} _{1}}, \ldots, \Sigma _{\mathrm{w} _{T}} )$. Furthermore, the matrices $\overline{\mathrm{A}}$ and $\overline{\mathrm{B}}$ have the following lower triangular and block-diagonal structures, respectively:

$$
\overline{\mathrm{A}} = \begin{bmatrix} \mathbb{I} & 0 & 0 & \ldots & 0 \\ 
\mathrm{A} _0 & \mathbb{I} & 0 & \ldots & \vdots \\ 
\mathrm{A} _1 \mathrm{A} _0 & \mathrm{A} _1 & \mathbb{I} & \ddots & \vdots \\ 
\vdots & \vdots &  \vdots & \ddots & 0 \\ 
\prod\limits _{i=0} ^{T-1} \mathrm{A} _{i} & \prod\limits _{i=1} ^{T-1} \mathrm{A} _{i} &  \ldots & \mathrm{A} _{T-1} & \mathbb{I}
\end{bmatrix}, \qquad 
\overline{\mathrm{B}} = \mathrm{diag} ( \mathbb{I}, \mathrm{B} _{0}, \ldots, \mathrm{B} _{T-1}).
$$

Using these structures, the `Bayesian4Wiener` library computes the matrix $\Psi^{\star}$, the vector $\psi^{\star}$, and estimates $\hat{\theta}_{\mathrm{B}}( \overline{\mathrm{y}} )$ if measurements are also provided.

- __Active learning__ seeks to develop input signals that further minimze estimation error. The optimal inputs can be determined independently of measurements, either a-priori or in real-time, by solving the following optimization problem:

$$
\overline{\mathrm{u}} ^{\star} \in \arg\min _{\overline{\mathrm{u}} \in \mathbb{U}} \mathbb{E} \left\[ \big\lVert \theta - \hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) \big\rVert ^{2} \right\] = \arg\min _{\overline{\mathrm{u}} \in \mathbb{U}} \mathcal{J} ^{\star} _{\mathrm{B}}( \overline{\mathrm{u}} ),
$$

where $\mathbb{U}$ represents the input space, which may impose physical constraints on feasible inputs for estimation, and $\mathcal{J} ^{\star} _{\mathrm{B}}( \overline{\mathrm{u}} )$ is the optimal estimation error obtained from the Bayesian affine estimator. The `Bayesian4Wiener` library solves this nonconvex optimization problem by applying the following adaptive stepsize projected gradient descent iteratively:

$$
\overline{\mathrm{u}} ^{k+1} = \mathcal{P} _{\mathbb{U}} \left\[ \overline{\mathrm{u}} ^{k} - \alpha _{k} \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) \right\],
$$

where $k$ represents the current iteration step, $\mathcal{P} _{\mathbb{U}} \[ \cdot \]$ denotes the projection operator that maps the argument onto $\mathbb{U}$, $\alpha _{k}$ is a positive stepsize, and $\nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k})$ is the gradient of the cost function evaluated at $\overline{\mathrm{u}}^{k}$. The library then solves the following to find the stepsize $\alpha _{k}$ at each iteration which requires initializations for parameters $\beta _{0} = \infty$ and $\alpha _{0} = 10^{-10}$:

$$
\alpha _{k} = \min \Biggl \\{ \sqrt{1+\beta _{k-1}}\alpha _{k-1}, \mkern2mu \frac{ \big\lVert \overline{\mathrm{u}} ^{k} - \overline{\mathrm{u}}^{k-1} \big\rVert }{ 2 \big\lVert \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) - \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k-1}) \big\rVert } \Biggr \\}, \qquad \beta _{k} = \frac{ \alpha _{k} }{ \alpha _{k-1} }, \qquad k \geq 1.
$$

Using the above algorithm, the `Bayesian4Wiener` library finds a locally opitmum $\overline{\mathrm{u}} ^{\star}$ which has less estimation error than a random input signal. For more information and detailed explanation, read our paper: [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490).

## Requirements

The `Bayesian4Wiener` library is currently available **only** in MATLAB and does not require any other dependencies:

- **Core Dependency**: MATLAB (R2020a or newer recommended)
- **Optional**: MATLAB Parallel Computing Toolbox
  - Enables parallel execution of `parfor` loops for accelerated computation.
  - Without this toolbox, `parfor` loops execute sequentially, which may increase processing time.

## Library
The `Bayesian4Wiener` library has three modes of operation in which either of the following computation is performed:  
  1) `estimatorOnly`: Solves the __optimal Bayesian MMSE affine estimator__ to find its parameters $\Psi^{\star}$ and $\psi^{\star}$. It further returns other underlying parameters computed for deriving the final estimator parameters such as those related to __Dynamic basis statistics__ (DBS), i.e., matrices $\overline{\Phi}$ and $\mathrm{M}$, as well as the optimal estimation error $\mathcal{J} ^{\star} _{\mathrm{B}}$ (For further detail, refer to Theorem 3.2 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490));
  2) `estimateTheta`: Solves the __optimal Bayesian MMSE affine estimator__ parameters and __theta estimates__ according to  $\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) = \Psi^{\star} \overline{\mathrm{y}} + \psi^{\star}$, using the measurements of the entire trajectory $\overline{\mathrm{y}}$. Similarly, it return $\Psi^{\star}$, $\psi^{\star}$, $\overline{\Phi}$, $\mathrm{M}$, $\mathcal{J} ^{\star} _{\mathrm{B}}$, and the theta estimates $\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} )$ (For further detail, refer to Theorem 3.2 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490));
  3) `activeLearning`: Finds the __optimal input__ sequence for the chosen horizon $T$ using the __active learning__ algorithm and computes its corresponding __optimal Bayesian MMSE affine estimator__ parameters. It returns the optimizer parameters such $\alpha$, $\beta$, a status flag demonstrating either it has converegd to a local minima or has reached the maximum iteration, the total number of iterations for the adaptive gradient descent algorithm, the vector of gradient and the final cost velue. It also returns the optimal input sequence $\overline{\mathrm{u}} ^{\star}$ as well as all the estimator parameters discussed in `estimatorOnly` mode (For further detail, refer to Theorem 3.2 and Section 5 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490)).

## Usage and examples

Here is how to use the `Bayesian4Wiener` library in your MATLAB project:
The `Bayesian4Wiener` library takes the following inputs:
  - **`model`**: This struct contains the following model parameters:
    - `numState`: The number of states $n _{\mathrm{x}}$;
    - `numInput`: The number of inputs $n _{\mathrm{u}}$;
    - `numTheta`: The number fo unknown parameters including zero frequency $N+1$;
    - `trajectoryT`: The time index of trajectory ending time $T$;
    - `matrixAbar`: The the state transition matrix for the entire trajectory $\overline{\mathrm{A}}$;
    - `matrixBbar`: The input matrix for the entire trajectory $\overline{\mathrm{B}}$;
    - `vecUbar`: The input vector for the entire trajectory $\overline{\mathrm{u}}$;
    - `allVecFreq`: The row vector collection of all frequencies $f _{n}$, **EXCLUDING FREQUENCY 0**, i.e., $\[ f _{1}, f _{2}, \ldots, f _{N} \]$;
    - `muTheta`: The mean of the unknown $\theta$ prior distribution, $\mu _{\theta}$;
    - `sigmaTheta`: The covariance of the unknown $\theta$ prior distribution, $\Sigma _{\theta}$;
    - `sigmaVbar`: The measurement noise covariance $\Sigma _{\mathrm{v}}$;
    - `sigmaWbar`: The process noise covariance $\Sigma _{\overline{\mathrm{w}}}$.
    
    All the above fields are mandatory and should be properly provided in order for the library to run.
    
  - **`settings`**: This struct contains the following fields:
  - - `mode`: One of the modes discussed in [Library](#requirements), i.e., $\\{$ `estimatorOnly`, `estimateTheta`, `activeLearning` $\\}$;
    - `verbose`: One of the verbosity levels $\\{ 0, 1, 2 \\}$;
      
    The above two fields should be set to either of the available options for the library to run.
    
    - `activeLearning`: is a struct with the following fields:
      - 'gradTol', 'costTol', 'maxIter', 'alpha', 'beta', ...
        'applyToInitX', 'existConstraint', 'vecUmax', 'vecUmin', ...
        'maxInitState', 'minInitState'
  - **`vecYbar`**:

such `alpha`, `beta`, a `status` flag demonstrating either it has converegd to a local minima `converged` or has reached the maximum iteration `max iterations`, `totalIter` showing the total number of iterations for the adaptive gradient descent algorithm, the vector of `gradient` and the final cost velue `optimalCost`. It also returns all the estimator parameters discussed in `estimatorOnly` mode.

    



     
