```@meta
CurrentModule = MarginalLogDensities
```

# MarginalLogDensities.jl
## Introduction

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

### Installation

MarginalLogDensities (MLD for short) requires Julia v1.10 or greater. It is a registered
Julia package, so you can install it from the REPL by typing `]` to enter package-manager
mode, then running

```julia-repl
pkg> add MarginalLogDensities
```

## Basic tutorial

A basic example follows. We start out by writing a function for our target log-
density. This function must have the signature `f(u, data)`, where `u` is a numeric vector
of parameters and `data` contains any data, or fixed parameters, needed by the function. 
The `data` argument can be anything you'd like, such as a `Vector`, `NamedTuple`,
`DataFrame`, or some other `struct`. (If your function does not depend on any external data,
just ignore that argument in the function body)

!!! note

    Note that the convention for this package is to write the function as a **positive** log-density.


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

```math
f_m(u_2) = \iint f(u) \; du_1 du_3.
```


After defining `marginal_logdensity`, you can call it just like the original function,
with the difference that you only need to supply `v`, the subset of parameters you're 
interested in, rather than the entire set `u`. 

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

## How it works

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

<!-- You also can re-run the same 
`MarginalLogDensity` with different `data` if you want (though if you're depending on 
the sparsity of your changing the `data`
causes the sparsity). -->

## A more realistic example: hierarchical regression

```@setup hr
using MarginalLogDensities
using Distributions
using StatsPlots
using Random

Random.seed!(123)
ncategories = 50
points_per_category = 5
categories = 1:ncategories
μ0 = 5.0
σ0 = 5.0
aa = rand(Normal(μ0, σ0), ncategories)
b = 4.5
σ = 1.5
category = repeat(categories, inner=points_per_category)
n = length(category)
x = rand(Uniform(-1, 1), n)
μ = [aa[category[i]] + b * x[i] for i in 1:n]
y = rand.(Normal.(μ, σ))

function loglik(u, data)
    # unpack the parameters
    μ0 = u[1]
    σ0 = exp(u[2])
    σ = exp(u[3])
    b = u[4]
    aa = u[5:end]
    # predict the data
    μ = [aa[data.category[i]] + b * data.x[i] for i in 1:data.n]
    # calculate and return the log-likelihood
    return loglikelihood(Normal(μ0, σ0), aa) + sum(logpdf.(Normal.(μ, σ), data.y))
end

data = (; x, y, category, n)
u0 = randn(4 + ncategories) 
iv = 1:4 
v0 = u0[iv]
iw = setdiff(eachindex(u0), iv)

mld = MarginalLogDensity(loglik, u0, iw, data)
```

Let's show how to use MLD on a slightly more complex problem, similar to one you might
actually encounter in practice: a hierarchical linear regression. Our response variable 
``y`` is a linear function of the predictor ``x`` plus some normally-distributed noise. 
Our data are divided into 50 different groups, and each group ``i`` has its own
intercept term ``a_i``. These intercepts are in turn drawn from a normal distribution with
mean ``\mu_0`` and standard deviation ``\sigma_0``.

```math
\begin{aligned}
\mu_{i, j} &= a_i + b x_{i,j} \\
y_{i,j} &\sim \mathrm{Normal}(\mu_{i, j}, \sigma) \\
a_i &\sim \mathrm{Normal}(\mu_0, \sigma_0)
\end{aligned}
```
Where ``i`` indexes the categories and ``j`` indexes the individual data points.

Choosing values for these parameters and simulating a dataset yields the following plot:

```@example
using MarginalLogDensities
using Distributions
using StatsPlots
using Random

Random.seed!(123)
ncategories = 50
points_per_category = 5
categories = 1:ncategories
μ0 = 5.0
σ0 = 5.0
aa = rand(Normal(μ0, σ0), ncategories)
b = 4.5
σ = 1.5
category = repeat(categories, inner=points_per_category)
n = length(category)
x = rand(Uniform(-1, 1), n)
μ = [aa[category[i]] + b * x[i] for i in 1:n]
y = rand.(Normal.(μ, σ))

scatter(x, y, color=category, legend=false, xlabel="x", ylabel="y")
```

Given this model structure, we can write a function for the log-likelihood of the data, 
conditional on a vector of parameters `u`. The function is written so that `u` is 
unbounded, that is, there are no constraints on any of its elements. This means it needs
to include `exp` transformations for the elements of `u` corresponding to`σ0` and `σ` to
make sure they end up non-negative inside the model.

```@example hr
function loglik(u, data)
    # unpack the parameters
    μ0 = u[1]
    σ0 = exp(u[2])
    σ = exp(u[3])
    b = u[4]
    aa = u[5:end]
    # predict the data
    μ = [aa[data.category[i]] + b * data.x[i] for i in 1:data.n]
    # calculate and return the log-likelihood
    return loglikelihood(Normal(μ0, σ0), aa) + sum(logpdf.(Normal.(μ, σ), data.y))
end

data = (; x, y, category, n)
u0 = randn(4 + ncategories) # μ0, σ0, σ, b, and aa
loglik(u0, data)
```

Say that we're not particularly interested in the individual intercepts ``a_i``, but want
to do inference on the other parameters. One way to approach this is to marginalize them

```@example hr
# select variables of interest out of complete parameter vector
iv = 1:4 
v0 = u0[iv]

# indices of nuisance parameters: everything esle 
iw = setdiff(eachindex(u0), iv)

# construct a MarginalLogDensity
mld = MarginalLogDensity(loglik, u0, iw, data)
```

We can then call `mld` like a function:

```@example hr
mld(v0)
```

And, using Optimization.jl, estimate the maximum marginal-likelihood values of our four
parameters of interest:



```@example hr
using Optimization, OptimizationOptimJL

opt_func = OptimizationFunction(mld)
opt_prob = OptimizationProblem(opt_func, v0, data)
opt_sol = solve(opt_prob, NelderMead())

μ0_opt, logσ0_opt, logσ_opt, b_opt = opt_sol.u

println("Estimated μ0:  $(round(μ0_opt, digits=2))  (true value $(μ0))")
println("Estimated σ0:  $(round(exp(logσ0_opt), digits=2))  (true value $(σ0))")
println("Estimated b:   $(round(b_opt, digits=2))  (true value $(b))")
println("Estimated σ:   $(round(exp(logσ_opt), digits=2))   (true value $(σ))")
```

## Using ComponentArrays

## Turing.jl integration

## Performance: Sparsity and AD

## API

```@index
```

```@autodocs
Modules = [MarginalLogDensities]
```