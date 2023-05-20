# MarginalLogDensities.jl

Tools for integrating out (marginalizing) parameters from log-probability functions,
such as likelihoods and posteriors of statistical models. This approach is
useful for models that incorporate random effects or latent variables that are 
important for structuring the model, but whose exact values may not be of
interest in and of themselves. (These are sometimes known as "nuisance parameters.")

A basic example is shown below. We start out by writing a function for our target log-
density. This function must have the signature `f(u, data)`, where `u` is a numeric vector
of parameters and `data` contains any data or fixed parameters. The `data` argument can
be any kind of object you'd like (a `Vector`, `NamedTuple`, `DataFrame`, etc.). 

> ℹ️ Note that this package's convention is to write this function as a **positive** log-density.


```julia
using MarginalLogDensities
data = ...

function loglik(u, data)
    ...
end
```

To calculate the 

```julia
u0 = ...
iw = ...
data = (...)
marginal_loglik = MarginalLogDensity(loglik, u0, ix, data)
```

Here `u` is a vector of all parameters, 
which is made up of both the parameters of interest `v` and the parameters to 
integrate out, `w`. (In the language of regression, these would be the fixed and 
random effects.) The vector `u0` provides an initial condition, and `iw` 
gives the indices of `w` inside `u`. The final argument is `data`. If your 
function does not depend on the `data` argument, it can be omitted, but
otherwise you need to pass it to `MarginalLogDensity` so it can 
set up its marginalization correctly.

You can then call the `MarginalLogDensity` object like the original function, with
the difference that you only need to supply `v`, the subset of parameters you're 
interested in, rather than the entire set `u`. The subset of `u` that you're
*not* explicitly interested in, `w`, is integrated out automatically.

```julia
initial_v = [...]
marginal_loglik(initial_v, data)
```

(As an aside, you also can re-run the same `MarginalLogDensity` with different `data`.)

The point of doing all this was probably to find an optimal set of parameters `u` for
your data, so let's do that. We load Optim.jl and call `optimize` on the anonymous
function `v -> marginal_loglik(v, data)`. By default, this uses the Nelder-Mead
algorithm.
```
using Optim
fit = optimize(v -> marginal_loglik(v, data), initial_v)
theta_hat = fit.minimizer
```