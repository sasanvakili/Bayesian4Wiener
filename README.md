# Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model
Consider a known discrete-time linear time-varying dynamical system where the states at time $t$ are observed through an _unknown_ observation model:

$$
\mathrm{x} _{t+1} = \mathrm{A} _{t} \mathrm{x} _{t} + \mathrm{B} _{t} \mathrm{u} _{t} + \mathrm{w} _{t+1},
$$
$$
\mathrm{y} _{t} = \sum\limits _{n=0}^{N} \theta _{n} \phi _{n}( \mathrm{x} _{t}) + \mathrm{v} _{t},
$$

where $t = \\{ 0, \\ldots, T \\}$, $\mathrm{x} _{t} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the vector of state variables, $\mathrm{A} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{x}} }$ is the state transition matrix, $\mathrm{u} _{t} \in \mathbb{R} ^{n _{\mathrm{u}}}$ is the vector of inputs, $\mathrm{B} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{u}}}$ is the input matrix, and $\mathrm{w} _{t+1} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the process noise. Observations are made through the scalar output measurements $\mathrm{y} _{t} \in \mathbb{R}$, while $\mathrm{v} _{t} \in \mathbb{R}$ represents the measurement noise. The _known_ basis functions $\phi _{n}( \mathrm{x} _{t})$ is the following Fourier basis function:

$$ 
\begin{cases}
\phi_ {0} ( \mathrm{x} _{t} ) = 1 & n = 0 \\
\phi _{n}( \mathrm{x} _{t} ) = \sum\limits _{\ell \in \\{ -1, 1 \\}} \mathrm{exp} ( j \langle { \ell f _{n} , \mathrm{x} _{t} } \rangle ) & n \geq 1,
\end{cases}
$$

with _known_ frequencies $f _{n} \in \mathbb{R} ^{n _{\mathrm{x}}}$, and $n$ denotes the frequency index.

- The __Optimal Bayesian Affine Estimator__ is of the form 

$$
\hat{\theta}_{\mathrm{B}}( \overline{\mathrm{y}} ) = \Psi^{\star} \overline{\mathrm{y}} + \psi^{\star},
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

where $\mathbb{U}$ represents the input space, which may impose physical constraints on feasible inputs for estimation, and $\mathcal{J} ^{\star} _{\mathrm{B}}( \overline{\mathrm{u}} )$ is the optimal estimation error obtained from the Bayesian affine estimator.


## Library
This library computes either of the following:  
  1) __Optimal Bayesian MMSE affine estimator__ parameters;
  2) __Optimal Bayesian MMSE affine estimator__ parameters and __theta estimates__;  
  3) __Optimal input__ using the __active learning__ algorithm and its corresponding __optimal Bayesian MMSE affine estimator__ parameters.

## Requirements

The `Bayesian4Wiener` library is currently available **only** in MATLAB and does not require any other dependencies:

- **Core Dependency**: MATLAB (R2020a or newer recommended)
- **Optional**: MATLAB Parallel Computing Toolbox
  - Enables parallel execution of `parfor` loops for accelerated computation.
  - Without this toolbox, `parfor` loops execute sequentially, which may increase processing time.


## Usage and examples

Here is how to use the `Bayesian4Wiener` library in your MATLAB project:



     
