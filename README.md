# Bayesian4Wiener  
This library is implemented according to the paper "Optimal Bayesian Affine Estimator and Active Learning for the Wiener Model".  
It computes either of the following:  
  1) Optimal Bayesian MMSE affine estimator parameters;  
  2) Optimal Bayesian MMSE affine estimator parameters and theta estimates;  
  3) Optimal input using the active learning algorithm and its corresponding optimal Bayesian MMSE affine estimator parameters.  

Consider a known discrete-time linear time-varying dynamical system where the states at time $t$ are observed through an _unknown_ observation model:  
$$
\mathrm{x}_{t+1} = \mathrm{A}_{t}\mathrm{x}_{t}+\mathrm{B}_{t}\mathrm{u}_t+\mathrm{w}_{t+1},
$$  
$$
\mathrm{y}_{t} = \!\sum_{n=0}^{N} \theta_{n} \phi_{n}(\mathrm{x}_{t}) + \mathrm{v}_{t}.
$$
