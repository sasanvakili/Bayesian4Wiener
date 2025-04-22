# Bayesian4Wiener: Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model

This library implements an optimal Bayesian affine estimator and active learning algorithm for a discrete-time linear time-varying dynamical system with an _unknown_ observation model. The system is described as follows:

$$
\mathrm{x} _{t+1} = \mathrm{A} _{t} \mathrm{x} _{t} + \mathrm{B} _{t} \mathrm{u} _{t} + \mathrm{w} _{t+1},
$$
$$
\mathrm{y} _{t} = \sum\limits _{n=0}^{N} \theta _{n} \phi _{n}( \mathrm{x} _{t}) + \mathrm{v} _{t} = \langle {\phi(\mathrm{x} _{t}), \theta} \rangle,
$$

where
*   $t = \\{ 0, \\ldots, T \\}$
*   $\mathrm{x} _{t} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the state vector.
*   $\mathrm{A} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{x}} }$ is the state transition matrix.
*   $\mathrm{B} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{u}}}$ is the input matrix.
*   $\mathrm{u} _{t} \in \mathbb{R} ^{n _{\mathrm{u}}}$ is the input vector.
*   $\mathrm{w} _{t+1} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the process noise.
*   $\mathrm{y} _{t} \in \mathbb{R}$ is the scalar output measurement.
*    $\mathrm{v} _{t} \in \mathbb{R}$  is the measurement noise.

The _known_ basis functions $\phi _{n}( \mathrm{x} _{t})$ are defined as the following Fourier bases:

$$ 
\begin{cases}
\phi_ {0} ( \mathrm{x} _{t} ) = 1 & n = 0 \\
\phi _{n}( \mathrm{x} _{t} ) = \sum\limits _{\ell \in \\{ -1, 1 \\}} \mathrm{exp} ( j \langle { \ell f _{n} , \mathrm{x} _{t} } \rangle ) & n \geq 1,
\end{cases}
$$

with _known_ frequencies $f _{n} \in \mathbb{R} ^{n _{\mathrm{x}}}$, where $n$ denotes the frequency index, and $\phi(\mathrm{x} _{t}) = \[ \phi _{0}(\mathrm{x} _{t}), \ldots, \phi _{N}(\mathrm{x} _{t}) \] ^{\mathsf{T}}$ is the vector of basis functions evaluated at $\mathrm{x} _{t}$.

## Optimal Bayesian Affine Estimator

The **optimal Bayesian affine estimator** has the form

$$
\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) = \Psi^{\star} \overline{\mathrm{y}} + \psi^{\star},
$$

This estimator computes the model parameters $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ from the measurement data $\mathrm{y} _{t}$ at all time steps, represented as $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$. To compute $\Psi^{\star}$ and $\psi^{\star}$, the models are represented in _lifted matrix_ form. The output function is

$$
\overline{\mathrm{y}} = \Phi ^{\mathsf{T}} \theta + \overline{\mathrm{v}},
$$

where
*   $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$ is the vector of measurements.
*   $\Phi = \[ \phi(\mathrm{x} _{0}), \ldots, \phi(\mathrm{x} _{T}) \]$ is the basis aggregation matrix.
*   $\theta = \[ \theta _{0}, \ldots, \theta _{N} \] ^{\mathsf{T}}$ is the vector of all _unknown_ parameters.
*   $\overline{\mathrm{v}} = \[ \mathrm{v} _{0}, \ldots, \\mathrm{v} _{T} \] ^{\mathsf{T}}$ is the measurement noise vector, where $\overline{\mathrm{v}} \sim \mathbb{P} ( 0, \Sigma _{\mathrm{v}})$ and $\Sigma _{\mathrm{v}} = \mathrm{diag}( \Sigma _{\mathrm{v} _{0}}^{2}, \ldots, \Sigma _{\mathrm{v} _{T}}^{2})$.

The prior information about $\theta$ is characterized by a probability distribution defined by its mean and covariance, $\mathbb{P} ( \mu _{\theta}, \Sigma _{\theta})$.

The process model is represented as

$$
\overline{\mathrm{x}} = \overline{\mathrm{A}} (\overline{\mathrm{B}} \overline{\mathrm{u}} + \overline{\mathrm{w}}),
$$

where
*   $\overline{\mathrm{x}} = \[ \mathrm{x} _{0} ^{\mathsf{T}}, \ldots, \mathrm{x} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the system states vector.
*   $\overline{\mathrm{u}} = \[ \mu _{\mathrm{x} _{0}} ^{\mathsf{T}}, \mathrm{u} _{0} ^{\mathsf{T}}, \ldots, \mathrm{u} _{T-1} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the input vector, with the mean of the initial state $\mu _{\mathrm{x} _{0}}$ as its first element.
*   $\overline{\mathrm{w}} = \[ \mathrm{w} _{0} ^{\mathsf{T}}, \mathrm{w} _{1} ^{\mathsf{T}}, \ldots, \mathrm{w} _{T} ^{\mathsf{T}} \] ^{\mathsf{T}}$ is the noise vector, where $\mathrm{w} _{0} \sim \mathbb{P} (0, \Sigma _{\mathrm{x} _{0}} )$ and $\overline{\mathrm{w}} \sim \mathbb{P} ( 0 , \Sigma _{\overline{\mathrm{w}}} )$. The covariance matrix is diagonal only if $\mathrm{w} _{t}$ are independent, i.e., $\Sigma _{\overline{\mathrm{w}}} = \mathrm{diag} ( \Sigma _{\mathrm{x} _{0}}, \Sigma _{\mathrm{w} _{1}}, \ldots, \Sigma _{\mathrm{w} _{T}} )$.

The matrices $\overline{\mathrm{A}}$ and $\overline{\mathrm{B}}$ have the following lower triangular and block-diagonal structures, respectively:

$$
\overline{\mathrm{A}} = \begin{bmatrix} \mathbb{I} & 0 & 0 & \ldots & 0 \\ 
\mathrm{A} _0 & \mathbb{I} & 0 & \ldots & \vdots \\ 
\mathrm{A} _1 \mathrm{A} _0 & \mathrm{A} _1 & \mathbb{I} & \ddots & \vdots \\ 
\vdots & \vdots &  \vdots & \ddots & 0 \\ 
\prod\limits _{i=0} ^{T-1} \mathrm{A} _{i} & \prod\limits _{i=1} ^{T-1} \mathrm{A} _{i} &  \ldots & \mathrm{A} _{T-1} & \mathbb{I}
\end{bmatrix}, \qquad 
\overline{\mathrm{B}} = \mathrm{diag} ( \mathbb{I}, \mathrm{B} _{0}, \ldots, \mathrm{B} _{T-1}).
$$

This library uses these structures to compute the matrix $\Psi^{\star}$, the vector $\psi^{\star}$, and estimates $\hat{\theta}_{\mathrm{B}}( \overline{\mathrm{y}} )$ if measurements are also provided.

## Active Learning

**Active learning** seeks to develop input signals that further minimize estimation error. The optimal inputs can be determined independently of measurements, either a-priori or in real-time, by solving the following optimization problem:

$$
\overline{\mathrm{u}} ^{\star} \in \arg\min _{\overline{\mathrm{u}} \in \mathbb{U}} \mathbb{E} \left\[ \big\lVert \theta - \hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) \big\rVert ^{2} \right\] = \arg\min _{\overline{\mathrm{u}} \in \mathbb{U}} \mathcal{J} ^{\star} _{\mathrm{B}}( \overline{\mathrm{u}} ),
$$

where
*   $\mathbb{U}$ represents the input space, which may impose physical constraints on feasible inputs for estimation.
*   $\mathcal{J} ^{\star} _{\mathrm{B}}( \overline{\mathrm{u}} )$ is the optimal estimation error obtained from the Bayesian affine estimator.

The `Bayesian4Wiener` library solves this nonconvex optimization problem by applying the following adaptive stepsize projected gradient descent iteratively:

$$
\overline{\mathrm{u}} ^{k+1} = \mathcal{P} _{\mathbb{U}} \left\[ \overline{\mathrm{u}} ^{k} - \alpha _{k} \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) \right\],
$$

where:

*   $k$ represents the current iteration step.
*   $\mathcal{P} _{\mathbb{U}} \[ \cdot \]$ denotes the projection operator that maps the argument onto $\mathbb{U}$.
*   $\alpha _{k}$ is a positive stepsize.
*   $\nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k})$ is the gradient of the cost function evaluated at $\overline{\mathrm{u}}^{k}$.

The library then solves the following to find the stepsize $\alpha _{k}$ at each iteration, which requires initializations for parameters $\beta _{0} = \infty$ and $\alpha _{0} = 10^{-10}$:

$$
\alpha _{k} = \min \Biggl \\{ \sqrt{1+\beta _{k-1}}\alpha _{k-1}, \mkern2mu \frac{ \big\lVert \overline{\mathrm{u}} ^{k} - \overline{\mathrm{u}}^{k-1} \big\rVert }{ 2 \big\lVert \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) - \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k-1}) \big\rVert } \Biggr \\}, \qquad \beta _{k} = \frac{ \alpha _{k} }{ \alpha _{k-1} }, \qquad k \geq 1.
$$

Using the above algorithm, the `Bayesian4Wiener` library finds a locally optimum $\overline{\mathrm{u}} ^{\star}$ which has less estimation error than a random input signal. 

For more information and a detailed explanation, refer to our paper: [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490).

## Requirements

The `Bayesian4Wiener` library is currently available **only** in MATLAB and has no other dependencies:

*   **Core Dependency**: MATLAB (R2020b or newer recommended)
*   **Optional**: MATLAB Parallel Computing Toolbox
    *   Invoke `ver('parallel')` in the MATLAB terminal command window to check its installation.
    *   Enables parallel execution of `parfor` loops for accelerated computation.
    *   Without this toolbox, `parfor` loops execute sequentially, which may increase processing time.

## Modes of Operation

The `Bayesian4Wiener` library has three modes of operation:

1.  `estimatorOnly`: Solves for the parameters $\Psi^{\star}$ and $\psi^{\star}$ of the **optimal Bayesian MMSE affine estimator**. It also returns underlying parameters used for deriving the final estimator parameters, such as those related to **Dynamic Basis Statistics** (DBS), i.e., matrices $\overline{\Phi}$ and $\mathrm{M}$, as well as the optimal estimation error $\mathcal{J} ^{\star} _{\mathrm{B}}$ (See Theorem 3.2 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490) for further details).

2.  `estimateTheta`: Solves for the parameters of the **optimal Bayesian MMSE affine estimator** and the **theta estimates** according to $\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} ) = \Psi^{\star} \overline{\mathrm{y}} + \psi^{\star}$, using the measurements of the entire trajectory $\overline{\mathrm{y}}$. It returns $\Psi^{\star}$, $\psi^{\star}$, $\overline{\Phi}$, $\mathrm{M}$, $\mathcal{J} ^{\star} _{\mathrm{B}}$, and the theta estimates $\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} )$ (See Theorem 3.2 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490) for further details).

3.  `activeLearning`: Finds the **optimal input** sequence for the chosen horizon $T$ using the **active learning** algorithm and computes its corresponding **optimal Bayesian MMSE affine estimator** parameters. It returns optimizer parameters such as $\alpha$, $\beta$, a status flag indicating convergence to a local minimum or reaching the maximum iteration, the total number of iterations for the adaptive gradient descent algorithm, the gradient vector, and the final cost value. It also returns the optimal input sequence $\overline{\mathrm{u}} ^{\star}$ as well as all the estimator parameters discussed in `estimatorOnly` mode (See Theorem 3.2 and Section 5 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490) for further details).

## Usage

The `Bayesian4Wiener` library can be invoked in MATLAB using the following command:

`[estimator, optimizer, optimalUbar, thetaEstimate] = Bayesian4Wiener(model, settings, vecYbar);`

Ensure that the the library's `src` folder is added to MATLAB's path (e.g., `addpath('./src')`), and the inputs are configured according to the descriptions below.

### Inputs

The `Bayesian4Wiener` library takes the following three major inputs:

*   **`model`**: This struct contains the following model parameters:
    *   `numState`: The number of states $n _{\mathrm{x}}$.
    *   `numInput`: The number of inputs $n _{\mathrm{u}}$.
    *   `numTheta`: The number of unknown parameters, including the zero frequency, $N+1$.  
    *   `trajectoryT`: The time index of the trajectory ending time $T$.
    *   `matrixAbar`: The state transition matrix for the entire trajectory, $\overline{\mathrm{A}}$.
    *   `matrixBbar`: The input matrix for the entire trajectory, $\overline{\mathrm{B}}$.
    *   `vecUbar`: The input vector for the entire trajectory, $\overline{\mathrm{u}}$.
    *   `allVecFreq`: The row vector collection of all frequencies $f_{n}$, **EXCLUDING FREQUENCY 0**, i.e., $\[ f _{1}, f _{2}, \ldots, f _{N} \]$.
    *   `muTheta`: The mean of the unknown $\theta$ prior distribution, $\mu _{\theta}$.
    *   `sigmaTheta`: The covariance of the unknown $\theta$ prior distribution, $\Sigma _{\theta}$.
    *   `sigmaVbar`: The measurement noise covariance, $\Sigma _{\mathrm{v}}$.
    *   `sigmaWbar`: The process noise covariance, $\Sigma _{\overline{\mathrm{w}}}$.

    All the above fields are mandatory and should be properly provided in order for the library to run.

*   **`settings`**: This struct contains the following fields:
    *   `mode`: One of the modes discussed in [Modes of Operation](#modes-of-operation), i.e., `estimatorOnly`, `estimateTheta`, or `activeLearning`.  
    *   `verbose`: One of the verbosity levels, where $0$ means silent execution and $1$ or $2$ provide minor debugging messages.  Allowed values are 0, 1, or 2.

    The above two fields should be set to one of the available options for the library to run.

    *   `activeLearning`: This struct contains the following fields:
        *   `gradTol`: Stopping criterion for gradient norm threshold, i.e., $\big\lVert \nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) \big\rVert <$ `gradTol`.
        *   `costTol`: Stopping criterion for cost decrease threshold, i.e., $\big\lvert \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k+1}) - \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{k}) \big\rvert <$ `costTol`.
        *   `maxIter`: Stopping criterion for the maximum number of iterations allowed. The algorithm will terminate when $k$ reaches `maxIter`.
        *   `alpha`: The initial value $\alpha _{0}$ required for the adaptive stepsize algorithm (recommended value: $\alpha _{0} = 10^{-10}$).
        *   `beta`: The initial value $\beta _{0}$ required for the adaptive stepsize algorithm (recommended value: $\beta _{0} = 10^{100}$).
        *   `applyToInitX`: A boolean variable determining whether or not the optimization of $\overline{\mathrm{u}}$ includes optimizing the initial state $\mu _{\mathrm{x} _{0}}$.
        *   `existConstraint`: A boolean variable determining whether or not there exists a constraint on the input $\mathbb{U}$.
        *   `vecUmax`: A vector with size $n _{\mathrm{u}}$ that contains the maximum values each dimension of $\mathrm{u} _{t}$ can take. This is required **ONLY** if `existConstraint` is set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.vecUmax = []`).
        *   `vecUmin`: A vector with size $n _{\mathrm{u}}$ that contains the minimum values each dimension of $\mathrm{u} _{t}$ can take. This is required **ONLY** if `existConstraint` is set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.vecUmin = []`).
        *   `maxInitState`: A vector with size $n _{\mathrm{x}}$ that contains the maximum values the initial state $\mu _{\mathrm{x} _{0}}$ can take. This is required **ONLY** if both `existConstraint` and `applyToInitX` are set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.maxInitState = []`).
        *   `minInitState`: A vector with size $n _{\mathrm{x}}$ that contains the minimum values the initial state $\mu _{\mathrm{x} _{0}}$ can take. This is required **ONLY** if both `existConstraint` and `applyToInitX` are set to true; otherwise, it can be left empty (i.e., `settings.activeLearning.minInitState = []`).

        All the above fields are required **ONLY** if the mode is set to `activeLearning` (i.e., `settings.mode = 'activeLearning'`); otherwise, `activeLearning` can be left empty (i.e., `settings.activeLearning = []`).

*   **`vecYbar`**: The vector of measurements, i.e., $\overline{\mathrm{y}} = \[ \mathrm{y} _{0}, \ldots, \mathrm{y} _{T} \] ^{\mathsf{T}}$. This is required **ONLY** if the mode is set to `estimateTheta`; otherwise, it can be left empty (i.e., `vecYbar = []`).

### Outputs

The `Bayesian4Wiener` library returns the following outputs, depending on the chosen mode of operation:

*   **`estimator`**: This struct contains the parameters of the Bayesian affine estimator and is always returned in all three modes (i.e., `estimatorOnly`, `estimateTheta` and `activeLearning`). It has the following fields:
    *   `matrixPhibar`: The matrix $\overline{\Phi}$ defined in Theorem 3.2 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490).
    *   `matrixM`: The matrix $\mathrm{M}$ defined in Theorem 3.2 of [Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model](https://arxiv.org/abs/2504.05490).
    *   `matrixPsi`: The estimator optimal parameter $\Psi^{\star}$.
    *   `vecPsi`: The estimator optimal parameter $\psi^{\star}$.
    *   `optimalErr`: The optimal estimation error $\mathcal{J} ^{\star} _{\mathrm{B}}$.

    All the above fields are returned for all three modes of operation.

*   **`optimizer`**: This struct contains the estimator parameters and results of the optimization process and is returned **ONLY** if the mode is set to `activeLearning`. It has the following fields:
    *   `alpha`: The final value of the stepsize $\alpha _{k}$ at termination.
    *   `beta`: The final value of $\beta _{k}$ at termination.
    *   `status`: A flag indicating whether the algorithm converged to a local minimum (`converged`) or reached the maximum number of iterations (`max iterations`).
    *   `totalIter`: The total number of iterations for the adaptive gradient descent algorithm.
    *   `gradient`: The gradient vector at termination, i.e., $\nabla _{\overline{\mathrm{u}}} \mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}}^{K})$.
    *   `optimalCost`: The final optimal cost value at termination, i.e., $\mathcal{J} ^{\star} _{\mathrm{B}}(\overline{\mathrm{u}} ^{\star})$.

    All the above fields are returned **ONLY** if the mode is set to `activeLearning`.

*   **`optimalUbar`**: The vector of the optimal input $\overline{\mathrm{u}} ^{\star}$. This is returned **ONLY** if the mode is set to `activeLearning`.

*   **`thetaEstimate`**: The vector of theta estimates $\hat{\theta} _{\mathrm{B}}( \overline{\mathrm{y}} )$. This is returned **ONLY** if the mode is set to `estimateTheta`.

### Examples

Examples for using each of the operation modes and setting various inputs are provided in the file `example.m` under the folder `example`. This example provides a comprehensive introduction to setting up and testing the `Bayesian4Wiener` library.

## Citing

If you use the `Bayesian4Wiener` library for research, please cite our accompanying paper:

```bibtex
@article{vakili2025optimal,
  title={Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model},
  author={Vakili, Sasan and Mazo Jr, Manuel and Mohajerin Esfahani, Peyman},
  journal={arXiv preprint arXiv:2504.05490},
  year={2025}
}
```
