```@meta
CurrentModule = MarginalLogDensities
```

# A more realistic example: hierarchical regression

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