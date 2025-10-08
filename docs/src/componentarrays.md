```@meta
CurrentModule = MarginalLogDensities
```

```@setup component
using MarginalLogDensities
using ComponentArrays
using Distributions
import ReverseDiff
using Optimization, OptimizationOptimJL
using StatsPlots
using LinearAlgebra
using Random

Random.seed!(1234)

A = [0.8 -0.2; -0.1 0.5]
b = 0.3
c = 0.1
n = 300

x = zeros(2, n)
for i in 2:n
    x[:, i] = A * x[:, i-1] + b * randn(2)
end
y = x .+ c .* randn.()

# plot(x', layout=(2,1), legend=false, ylabel=["x1" "x2"], xlabel=["" "Time"])
# scatter!(y', markersize=2, markerstrokewidth=0)

function logdensity(u, data)
    A = u.A
    b = exp(u.log_b)
    x  = u.x

    y = data.y
    n = data.n
    c  = data.c

    # Prior on x[1]
    ll = logpdf(MvNormal(zeros(2), 10), x[:, 1])
    # first observation
    ll += logpdf(MvNormal(x[:, 1], c), y[:, 1])
    # rest of time series
    for t in 2:n
        ll += logpdf(MvNormal(A * x[:, t-1], b), x[:, t])
        ll += logpdf(MvNormal(x[:, t], c), y[:, t])
    end
    return ll
end

u0 = ComponentArray(
    A = A,
    log_b = 0,
    x = ones(2, n)
)
data = (y = y, n = n, c = c)
joint_vars = [:A, :log_b]

mld = MarginalLogDensity(logdensity, u0, [:x], data,
    LaplaceApprox(adtype=AutoReverseDiff()), sparsity_detector=TracerSparsityDetector())
v0 = u0[joint_vars]
# @time mld(v0)
func = OptimizationFunction(mld);
prob = OptimizationProblem(func, v0, data);
sol = solve(prob, NelderMead())
```

# Making life easier with ComponentArrays

MarginalLogDensities requires you to write your log-density as a function of a single flat
array of parameters, `u`. This makes the internal marginalization calculations easier to
perform, but as your model becomes more complex, it becomes more annoying and error-prone
to keep track of which indices in `u` refer to which variables inside your model. One 
solution to this problem is provided in [ComponentArrays.jl](https://github.com/SciML/ComponentArrays.jl/tree/main),
from the SciML ecosystem.

A `ComponentArray` is a type that behaves like an ordinary `Array`, but organized into 
blocks that can be accessed via named indices:

```julia-repl
kjulia> using ComponentArrays

julia> u = ComponentArray(a = 5.0, b = [-0.1, 4.8])
ComponentVector{Float64}(a = 5.0, b = [-0.1, 4.8])

julia> u.a
5.0

julia> u.b
2-element view(::Vector{Float64}, 2:3) with eltype Float64:
 -0.1
  4.8

julia> ones(3,3) * u # behaves like a normal vector
3-element Vector{Float64}:
 9.7
 9.7
 9.7
```

If you define your parameters as a `ComponentVector`, working with them inside your
log-density function becomes much easier, without introducing any computational overhead.

MarginalLogDensities works with `ComponentVectors` as well: instead of specifying the 
integer indices of the variables to marginalize, you can give `Symbol`s referring to 
blocks of the parameter vector.

To illustrate this, we'll fit a state-space model to some simulated time-series data.
Specifically, we will have a two-dimensional vector autoregressive process, where the 
hidden state $\mathbf{x}$ evolves according to 

```math
\begin{aligned}
\mathbf{x}_t &= A \mathbf{x}_{t-1} + \mathbf{\epsilon}_t \\
\mathbf{\epsilon} $\sim \mathrm{MvNormal}(0, b^2 I)
\end{aligned}
```
where $A$ is a square matrix and $mathbf{\epsilon}$ is the process noise, with standard 
deviation $b$. Out data, $\mathbf{y}_t$, are centered on $\mathbf{y}_t$, plus some 
Gaussian noise with standard deviation $c$.

```math
\mathbf{y}_t \sim \mathrm{MvNormal(\mathbf{x}_t, c^2 I)}
```

We'll assume we know $c$ a priori, but would like to estimate the transition matrix $A$ 
and the process noise $b$, while integrating out the unobserved states $\mathbf{x}_t$ for 
all times $t$. 

Simulating and plotting our time series and observations:

```@example component
using MarginalLogDensities
using ComponentArrays
using Distributions
import ReverseDiff
using Optimization, OptimizationOptimJL
using StatsPlots
using LinearAlgebra
using Random

Random.seed!(1234)

A = [0.8 -0.2; 
    -0.1  0.5]
b = 0.3
c = 0.1

x = zeros(2, n)
for i in 2:n
    x[:, i] = A * x[:, i-1] + b * randn(2)
end
y = x .+ c .* randn.()

plot(x', layout=(2,1), legend=false, ylabel=["x1" "x2"], xlabel=["" "Time"])
scatter!(y', markersize=2, markerstrokewidth=0)
```

Next, we define our log-density function. Note how we unpack the parameters using dot-
notation, e.g. `log_b = u.log_b` (where `log_b` is a scalar) and `A = u.A` (where `A` is a
vector).

The parameters for the variances `b` and `c` are supplied in the log-domain and `exp`-
transformed to make sure they're always positive inside the model.

```@example component

function logdensity(u, data)
    A = u.A
    b = exp(u.log_b)
    x  = u.x

    y = data.y
    n = data.n
    c  = data.c

    # Prior on initial state
    ll = logpdf(MvNormal(zeros(2), 10), x[:, 1])
    # first observation
    ll += logpdf(MvNormal(x[:, 1], c), y[:, 1])
    # rest of time series
    for t in 2:n
        ll += logpdf(MvNormal(A * x[:, t-1], b), x[:, t])
        ll += logpdf(MvNormal(x[:, t], c), y[:, t])
    end
    return ll
end
```

The inital value for our parameter vector is constructed as a `ComponentArray` to make it
compatible with `logdensity` as it's written. The fixed `data` are a `NamedTuple`. These 
are passed to the `MarginalLogDensity` constructor along with the function, as usual.


```@example component
u0 = ComponentArray(
    A = A,
    log_b = 0,
    x = ones(2, n)
)
data = (y = y, n = n, c = c)
joint_vars = [:A, :log_b]

mld = MarginalLogDensity(logdensity, u0, [:x], data,
    LaplaceApprox(adtype=AutoReverseDiff()), sparsity_detector=TracerSparsityDetector())
```

However, we've specified we want to integrate out the latent states `x` as a 
random effect: we just pass a `Vector` containing the symbol(s) of the variables to 
marginalize.

!!! note 

    We specify the ReverseDiff AD backend to use for gradients inside the
    `LaplaceApproximation` method. Since our marginal variables are fairly high dimensional 
    (a 1D latent state times 300 time steps gives 600 marginal parameters), reverse-mode AD is
    much faster than the default ForwardDiff backend. See the page on sparse automatic 
    differentiation for more info.

From here, we select the subset of `u0` containing the non-marginal variables, set up an
optimization problem based on `mld`, and solve it.

```@example component
joint_vars = [:A, :log_b]
v0 = u0[joint_vars]
func = OptimizationFunction(mld);
prob = OptimizationProblem(func, v0, data)
sol = solve(prob, NelderMead())
```

We can use the delta method (based on finite differences) to estimate standard errors for
our time-series parameters.

```@example component
estimates = sol.u
std_err = 1.96 ./ sqrt.(diag(hessian(v -> -mld(v), AutoFiniteDiff(), sol.u)))
scatter(Vector(sol.u), yerr = std_err, label="Fitted",
    xticks=(1:5, ["A[1,1]", "A[1,2]", "A[2,1]", "A[2,2]", "log_b"]))
scatter!([vec(A); log(b)], label="True value")
``` 

The plot shows that they match up fairly well with the true values.

We can also access the latest value of the latent state, conditional on these point 
estimates:

```@example component
mld(sol.u)
x_hat = cached_params(mld).x 
x_err = 1.96 ./ sqrt.(diag(cached_hessian(mld)))

plot(x_hat', ribbon=x_err', label="Estimated state ± 2 s.d.", layout=(2, 1), color=3)
plot!(x', label="Latent state", xlabel="Time step", color=1)
scatter!(y', label="Noisy observations", markerstrokewidth=0, markersize=2, color=2)
```