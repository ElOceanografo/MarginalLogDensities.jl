
```@meta
CurrentModule = MarginalLogDensities
```

# Turing.jl integration

Starting with Turing [v0.40.4](https://github.com/TuringLang/Turing.jl/releases/tag/v0.40.4)
 / DynamicPPL [v0.37.4](https://github.com/TuringLang/DynamicPPL.jl/releases/tag/v0.37.4),
it is possible to construct a `MarginalLogDensity` directly from a Turing model, 
specifying the variables to marginalize by their names.

## Model definition

Models can be defined using Turing's `@model` macro and probabilistic programming syntax. 
Here, we define a simple model with a latent variable `x` that follows a Gaussian 
distribution with mean `m` and covariance `C`.

```@setup turing
using Turing
using MarginalLogDensities
using DynamicPPL
using Distributions
using LinearAlgebra
using StatsPlots
using Random

Random.seed!(4321)

r = 0.4
a = 2
n = 50
m = fill(a, n)
C = Tridiagonal(fill(-r, n-1), ones(n), fill(-r, n-1))
y = rand(MvNormal(m, C))
plot(y)

@model function demo(y)
    n = length(y)
    r ~ Uniform(-0.5, 0.5)
    a ~ Normal(0, 5)
    m ~ MvNormal(fill(a, n), 1)
    C = Tridiagonal(fill(-r, n-1), ones(n), fill(-r, n-1))

    y ~ MvNormal(m, C)
end

full = demo(y)
marginal = marginalize(full, [@varname(m)], 
    sparsity_detector=DenseSparsityDetector(AutoForwardDiff(), atol=1e-9))
njoint(marginal)
marginal([0.1, 1.0])

using Optimization, OptimizationOptimJL

v0 = zeros(2)
opt_func = OptimizationFunction(marginal)
opt_prob = OptimizationProblem(opt_func, v0, ())
opt_sol = solve(opt_prob, NelderMead())
```

```@example turing
using Turing
using MarginalLogDensities
using DynamicPPL
using Distributions
using LinearAlgebra
using StatsPlots
using Random

Random.seed!(4321)

r = 0.4
a = 2
n = 50
m = fill(a, n)
C = Tridiagonal(fill(-r, n-1), ones(n), fill(-r, n-1))
y = rand(MvNormal(m, C))
plot(y)

@model function demo(y)
    n = length(y)
    r ~ Uniform(-0.5, 0.5)
    a ~ Normal(0, 5)
    m ~ MvNormal(fill(a, n), 1)
    C = Tridiagonal(fill(-r, n-1), ones(n), fill(-r, n-1))

    y ~ MvNormal(m, C)
end

full = demo(y)
marginal = marginalize(full, [@varname(m)],
    sparsity_detector=DenseSparsityDetector(AutoForwardDiff(), atol=1e-9))

njoint(marginal)
marginal([0.1, 1.0])
```

## Maximum a-posteriori optimization

```@example turing
using Optimization, OptimizationOptimJL

v0 = zeros(2)
opt_func = OptimizationFunction(marginal)
opt_prob = OptimizationProblem(opt_func, v0, ())
opt_sol = solve(opt_prob, NelderMead())
```

## MCMC Sampling

Sampling from a `MarginalLogDensity` is currently a bit more awkward than doing so from a
Turing model, but not too bad. Keep in mind that `MarginalLogDensity` objects don't
currently work with AD, so samplers must either be gradient free (like random-walk
Metropolis Hastings), or use a finite-difference backend (e.g. `AutoFiniteDiff()`.)

The code below was adapted from @torfjelde's example on GitHub
[here](https://github.com/TuringLang/Turing.jl/issues/2398#issuecomment-2514212264). We 
request 100 samples with a thinning rate of 20 - that is, we'll run 2,000 samples and keep 
every 20th one. We also set the inital parameters to the MAP optimum we found before, 
which speeds up convergence in this case.

```@example turing
using AbstractMCMC, AdvancedMH

sampler = AdvancedMH.RWMH(njoint(marginal))
chain = sample(marginal, sampler, 100; chain_type=MCMCChains.Chains,
    thinning=20, initial_params=opt_sol.u,
    # HACK: this a dirty way to extract the variable names in a model; it won't work in general.
    param_names=setdiff(keys(DynamicPPL.untyped_varinfo(full)), [@varname(m)])
)
plot(chain)
```

These chains are short for the sake of the demo, but could easily be run longer to get a 
smoother posterior.

Sampling using Hamiltonian Monte Carlo is possible, but is currently a bit more awkward.
The following is adapted from the AdvancedHMC 
[documentation](https://turinglang.org/AdvancedHMC.jl/stable/get_started/):

```@example turing
using AdvancedHMC
import FiniteDiff
using LogDensityProblems

# Set the number of samples to draw and warmup iterations
n_samples, n_adapts = 200, 100

# Define a Hamiltonian system
metric = DiagEuclideanMetric(njoint(marginal))
hamiltonian = Hamiltonian(metric, marginal, AutoFiniteDiff())
initial_ϵ = 0.4 
integrator = Leapfrog(initial_ϵ)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

samples, stats = sample(
    hamiltonian, kernel, opt_sol.u, n_samples, adaptor, n_adapts; progress=true,
)
plot(hcat(samples...)', layout=(2,  1), xlabel="Iteration", ylabel=["r" "a"], legend=false)
```

## Un-linking parameters

One of the nice things about using Turing is that the DynamicPPL modeling language handles
all the variable transformations, so that optimization and sampling can take place in 
unconstrained parameter space even when you have bounded parameters (e.g. `r` in this 
example). By default, `marginalize` sets up the `MarginalLogDensity` to use these
unconstrained (a.k.a. "linked", to use the language familiar from generalized linear
modeling) parameters. 

If you want transform them back to unlinked space, i.e. how they appear inside the model, 
you need to construct a `VarInfo` and query it like this:

```@example turing
vi = VarInfo(marginal, opt_sol.u)
vi[@varname(r)]
```