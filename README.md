# Bayesian4Wiener  
This library is implemented according to the paper "__Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model__". It computes either of the following:  
  1) __Optimal Bayesian MMSE affine estimator__ parameters;
  2) __Optimal Bayesian MMSE affine estimator__ parameters and __theta estimates__;  
  3) __Optimal input__ using the __active learning__ algorithm and its corresponding __optimal Bayesian MMSE affine estimator__ parameters.
     
Consider a known discrete-time linear time-varying dynamical system where the states at time $t$ are observed through an _unknown_ observation model:

$$
\mathrm{x} _{t+1} = \mathrm{A} _{t} \mathrm{x} _{t} + \mathrm{B} _{t} \mathrm{u} _{t} + \mathrm{w} _{t+1},
$$

$$
\mathrm{y} _{t} = \sum\limits _{n=0}^{N} \theta _{n} \phi _{n}( \mathrm{x} _{t}) + \mathrm{v} _{t},
$$

where $t = \\{ 0, \\ldots, T \\}$ represents the time index starting from $0$ and ending at time $T$, $\mathrm{x} _{t} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the vector of state variables, $\mathrm{A} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{x}} }$ is the state transition matrix, $\mathrm{u} _{t} \in \mathbb{R} ^{n _{\mathrm{u}}}$ is the vector of inputs, $\mathrm{B} _{t} \in \mathbb{R} ^{n _{\mathrm{x}} \times n _{\mathrm{u}}}$ is the input matrix, and $\mathrm{w} _{t+1} \in \mathbb{R} ^{n _{\mathrm{x}}}$ is the process noise. 

Observations are made through the scalar output measurements $\mathrm{y} _{t} \in \mathbb{R}$, while $\mathrm{v} _{t} \in \mathbb{R}$ represents the measurement noise. The _known_ basis functions $\phi _{n}( \mathrm{x} _{t})$ is the following Fourier basis function:

$$ 
\begin{cases}
\phi_ {0} ( \mathrm{x} _{t} ) = 1 & n = 0 \\
\phi _{n}( \mathrm{x} _{t} ) = \sum\limits _{\ell \in \\{ -1, 1 \\}} \mathrm{exp} ( j \langle { \ell f _{n} , \mathrm{x} _{t} } \rangle ) & n \geq 1,
\end{cases}
$$

with _known_ frequencies $f _{n} \in \mathbb{R} ^{n _{\mathrm{x}}}$, and $n$ denotes the frequency index.
