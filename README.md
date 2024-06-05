# MarginalLogDensities.jl

[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)](https://github.com/ElOceanografo/MarginalLogDensities.jl)


[![Build Status](https://github.com/ElOceanografo/MarginalLogDensities.jl/workflows/CI/badge.svg)](https://github.com/ElOceanografo/MarginalLogDensities.jl/actions?query=workflow%CI+branch%3Amaster)

This package implements tools for integrating out (marginalizing) parameters
from log-probability functions, such as likelihoods and posteriors of statistical
models. This approach is useful for fitting models that incorporate random effects or 
latent variables that are important for structuring the model, but whose exact 
values may not be of interest in and of themselves. In the language of regression, 
these are known as "random effects," and in other contexts, they may be called "nuisance
parameters." Whatever they are called, we hope that our log-density function can be
optimized faster and/or more reliably if we focus on the parameters we
are *actually* interested in, averaging the log-density over all possible values of 
the ones we are not.

A basic example is shown below. We start out by writing a function for our target log-
density. This function must have the signature `f(u, data)`, where `u` is a numeric vector
of parameters and `data` contains any data, or fixed parameters, needed by the function. 
The `data` argument can be anything you'd like, such as a `Vector`, `NamedTuple`,
`DataFrame`, or some other `struct`. (If your function does not depend on any external data,
just ignore that argument in the function body)

> ⚠️ Note that the convention for this package is to write the function as a **positive** log-density.


```julia
using MarginalLogDensities, Distributions, LinearAlgebra

N = 3
σ = 1.5
dist = MvNormal(σ^2 * I(3))
data = (N=N, dist=dist)

function logdensity(u, data)
   return logpdf(data.dist, u) 
end
```

We then set up a marginalized version of this function. First, we set an initial
value for our parameter vector `u`, as well as a set of indices `iw` indicating
the subset of `u` that we want to integrate out. 

```julia
u = [1.1, -0.3, 0.1] # arbitrary initial values
iw = [1, 3]
```

We create the marginalized version of the function by calling the `MarginalLogDensity`
constructor with the original function `logdensity`, the full parameter vector `u`, the
indices of the marginalized values `iw`, and the `data`. (If your function does not depend on
the `data` argument, it can be omitted here.)

```julia
marginal_logdensity = MarginalLogDensity(logdensity, u, iw, data)
```

Here, we are saying that we want to calculate `logdensity` as a function of `u[2]` only, 
while integrating over all possible values of `u[1]` and `u[3]`. In the laguage of 
mathematics, if we write `u -> logdensity(u, data)` as $f(u)$, then the marginalized 
function `v -> marginal_logdensity(v, data)` is calculating

$$
f_m(u_2) = \int \int f(u) \; du_1 du_2.
$$

By default, this package uses Laplace's method to approximate this integral. The Laplace approximation
is fast in high dimensions, and works well for log-densities that are approximately Gaussian. The
marginalization can also be done via numerical integration, a.k.a. cubature, which may be more accurate
but will not scale well for higher dimensional problems. You can choose which marginalizer to use by passing the
appropriate `AbstractMarginalizer` to `MarginalLogDensity`:

```julia
MarginalLogDensity(logdensity, u, iw, data, Cubature())
MarginalLogDensity(logdensity, u, iw, data, LaplaceApprox())
```

Both `Cubature()` and `LaplaceApprox()` can be specified with various options; refer to their 
docstrings for details.

After defining `marginal_logdensity`, you can call it just like the original function,
with the difference that you only need to supply `v`, the subset of parameters you're 
interested in, rather than the entire set `u`. You also can re-run the same 
`MarginalLogDensity` with different `data` if you want (though if you're depending on 
the sparsity of your changing the `data`
causes the sparsity).

```julia
initial_v = [1.0] # another arbitrary starting value
length(initial_v) == N - length(iw) # true
marginal_logdensity(initial_v, data)
# -1.5466258635350596
```
Compare this value to the analytic marginal density, i.e. the log-probability of
$N(0, 1.5)$ at 1.0:
```julia
logpdf(Normal(0, 1.5), 1.0)
# -1.5466258635350594
```

The point of doing all this was to find an optimal set of parameters `v` for
your data. This package includes an interface to Optimization.jl that 
works directly with a `MarginalLogDensity` object, making  optimization easy. The simplest
way is to construct an `OptimizationProblem` directly from the `MarginalLogDensity` and
`solve` it:

```julia
using Optimization, OptimizationOptimJL

opt_problem = OptimizationProblem(marginal_logdensity, v0)
opt_solution = solve(opt_problem, NelderMead())
```

If you want more control over options, for instance setting an AD backend, you can
construct an `OptimizationFunction` explicitly:

```julia
opt_function = OptimizationFunction(marginal_logdensity, AutoFiniteDiff())
opt_problem = OptimizationProblem(opt_function, v0)
opt_solution = solve(opt_problem, LBFGS())
```

Note that at present we can't differentiate through the Laplace approximation, so outer 
optimizations like this need to either use a gradient-free solver (like `NelderMead()`),
or a finite-difference backend (like `AutoFiniteDiff()`). This is on the list of planned
improvements.

A more realistic application to a mixed-effects regression can be found in this
[example script](https://github.com/ElOceanografo/MarginalLogDensities.jl/blob/master/examples/example.jl).