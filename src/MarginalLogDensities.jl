module MarginalLogDensities

using Reexport
using Optimization
using OptimizationOptimJL
import ForwardDiff
@reexport using DifferentiationInterface
@reexport using ADTypes
@reexport using SparseConnectivityTracer
@reexport using SparseMatrixColorings
using LinearAlgebra
using SparseArrays
using ComponentArrays
using ChainRulesCore
using HCubature

export MarginalLogDensity,
    AbstractMarginalizer,
    LaplaceApprox,
    Cubature,
    dimension,
    imarginal,
    ijoint,
    nmarginal,
    njoint,
    cached_hessian,
    merge_parameters,
    split_parameters,
    optimize_marginal!
    # hessdiag

abstract type AbstractMarginalizer end

"""
    `LaplaceApprox([solver=LBFGS() [; adtype=AutoForwardDiff(), opt_func_kwargs...]])

Construct a `LaplaceApprox` marginalizer to integrate out marginal variables via
the Laplace approximation. This method will usually be faster than `Cubature`, especially
in high dimensions, though it may not be as accurate.

# Arguments
- `solver=LBFGS()` : Algorithm to use when performing the inner optimization to find the
mode of the marginalized variables. Can be any algorithm defined in Optim.jl.
- `adtype=AutoForwardDiff()` : Automatic differentiation type to use for the inner
optimization, specified via the ADTypes.jl interface. `AutoForwardDiff()` is robust and
fast for small problems; for larger ones `AutoReverseDiff()` or `AutoZygote()` are likely
better.
- `opt_func_kwargs` : Optional keyword arguments passed on to
`Optimization.OptimizationFunction`.
"""
struct LaplaceApprox{TA, TT, TS} <: AbstractMarginalizer
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
end

function LaplaceApprox(solver=LBFGS(); adtype=AutoForwardDiff(),
    opt_func_kwargs...)
    return LaplaceApprox(solver, adtype, opt_func_kwargs)
end

"""
    Cubature([; solver=LBFGS(), adtype=AutoForwardDiff(),
        upper=nothing, lower=nothing, nσ=6; opt_func_kwargs...])

Construct a `Cubature` marginalizer to integrate out marginal variables via
numerical integration (a.k.a. cubature).

If explicit upper and lower bounds for the integration are not supplied, this marginalizer
will attempt to select good ones by first optimizing the marginal variables, doing a 
Laplace approximation at their mode, and then going `nσ` standard deviations away
on either side, assuming approximate normality.

The integration is performed using `hcubature` from Cubature.jl.

# Arguments
- `solver=LBFGS()` : Algorithm to use when performing the inner optimization to find the
mode of the marginalized variables. Can be any algorithm defined in Optim.jl.
- `adtype=AutoForwardDiff()` : Automatic differentiation type to use for the 
inner optimization. `AutoForwardDiff()` is robust and fast for small problems; for larger
ones `AutoReverseDiff()` or `AutoZygote()` are likely better.
- `upper`, `lower` : Optional upper and lower bounds for the numerical integration. If
supplied, they must be numeric vectors the same length as the marginal variables.
- `nσ=6.0` : If `upper` and `lower` are not supplied, integrate this many standard
deviations away from the mode based on a Laplace approximation to the curvature at that 
point.
- `opt_func_kwargs` : Optional keyword arguments passed on to
`Optimization.OptimizationFunction`.
"""
struct Cubature{TA, TT, TS, TV, T} <: AbstractMarginalizer
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
    upper::TV
    lower::TV
    nσ::T
end

function Cubature(; solver=LBFGS(), adtype=AutoForwardDiff(), 
    upper=nothing, lower=nothing, nσ=6.0, opt_func_kwargs...)
    return Cubature(solver, adtype, opt_func_kwargs, promote(upper, lower)..., nσ)
end


"""
    `MarginalLogDensity(logdensity, u, iw, data, [method=LaplaceApprox(); 
    [hess_adtype=nothing, sparsity_detector=DenseSparsityDetector(method.adtype, atol=cbrt(eps())),
    coloring_algorithm=GreedyColoringAlgorithm()]])`

Construct a callable object which wraps the function `logdensity` and
integrates over a subset of its arguments.

The resulting `MarginalLogDensity` object  `mld` can then be called like a function
as `mld(v, data)`, where `v` is the subset of the full parameter vector `u` which is
*not* indexed by `iw`.  If `length(u) == n` and `length(iw) == m`, then `length(v) == n-m`.

# Arguments
- `logdensity` : function with signature `(u, data)` returning a positive
log-probability (e.g. a log-pdf, log-likelihood, or log-posterior). In this
function, `u` is a vector of variable parameters and `data` is an object (Array,
NamedTuple, or whatever) that contains data and/or fixed parameters.
- `u` : Vector of initial values for the parameter vector.
- `iw` : Vector of indices indicating which elements of `u` should be marginalized.
- `data=()` : Optional argument
- `method` : How to perform the marginalization.  Defaults to `LaplaceApprox()`; `Cubature()`
is also available.
- `hess_adtype = nothing` : Specifies how to calculate the Hessian of the marginalized 
variables. If not specified, defaults to a sparse second-order method using forward AD 
over the AD type given in the `method` (`AutoForwardDiff()` is the default). 
Other backends can be set by loading the appropriate AD package and using the ADTypes.jl 
interface.
- `sparsity_detector = DenseSparsityDetector(method.adtype, atol=cbrt(eps))` : How to
perform the sparsity detection. Detecting sparsity takes some time and may not be worth it
for small problems, but for larger problems it can be extremely worth it. The default 
`DenseSparsityDetector` is most robust, but if it's too slow, or if you're running out of 
memory on a larger problem, try the tracing-based dectectors from SparseConnectivityTracer.jl.
- `coloring_algorithm = GreedyColoringAlgorithm()` : How to determine the matrix "colors"
to compress the sparse Hessian.


# Examples
```julia-repl
julia> using MarginalLogDensities, Distributions

julia> N = 4

julia> dist = MvNormal(I(3))

julia> data = (N=N, dist=dist)

julia> function logdensity(u, data) # arbitrary simple density function
           return logpdf(data.dist, u) 
       end

julia> u0 = rand(N)

julia> mld = MarginalLogDensity(logdensity, u0, [1, 3], data)

julia> mld(rand(2), data)

```
"""
struct MarginalLogDensity{
        TF, 
        TU<:AbstractVector, 
        TD, 
        TV<:AbstractVector, 
        TW<:AbstractVector, 
        TM<:AbstractMarginalizer,
        TF1<:OptimizationFunction, 
        TP<:OptimizationProblem,
        TC<:OptimizationCache,
        TH<:AbstractMatrix,
        TB<:ADTypes.AbstractADType,
        TE
}
    logdensity::TF
    u::TU
    data::TD
    iv::TV
    iw::TW
    method::TM
    f_opt::TF1
    prob::TP
    cache::TC
    H::TH
    hess_adtype::TB
    hess_prep::TE
end

function MarginalLogDensity(logdensity, u, iw, data=(), method=LaplaceApprox(); 
        hess_adtype=nothing, sparsity_detector=DenseSparsityDetector(method.adtype, atol=sqrt(eps())),
        coloring_algorithm=GreedyColoringAlgorithm())
    iv = setdiff(eachindex(u), iw)
    w = u[iw]
    v = u[iv]
    p2 = (p=data, v=v)
    f(w, p2) = -logdensity(merge_parameters(p2.v, w, iv, iw), p2.p)
    f_opt = OptimizationFunction(f, method.adtype; method.opt_func_kwargs...)
    prob = OptimizationProblem(f_opt, w, p2)
    cache = init(prob, method.solver)
    
    if isnothing(hess_adtype)
        hess_adtype = AutoSparse(
            SecondOrder(AutoForwardDiff(), method.adtype),
            sparsity_detector,
            coloring_algorithm
        ) 
    end
    prep = prepare_hessian(f, hess_adtype, w, Constant(p2))
    H = hessian(f, prep, hess_adtype, w, Constant(p2))
    return MarginalLogDensity(logdensity, u, data, iv, iw, method, f_opt, prob, cache,
        H, hess_adtype, prep)
end

function MarginalLogDensity(logdensity, u::ComponentArray, iw::Vector{Symbol},
        args...; kwargs...)
    iw1 = reduce(vcat, label2index(u, label) for label in iw)
    u1 = Vector(u)
    MarginalLogDensity(logdensity, u1, iw1, args..., kwargs...)
end

function Base.show(io::IO, mld::MarginalLogDensity)
    T = typeof(mld.method).name.name
    str = "MarginalLogDensity of function $(repr(mld.logdensity))\nIntegrating $(length(mld.iw))/$(length(mld.u)) variables via $(T)"
    write(io, str)
end

function (mld::MarginalLogDensity)(v::AbstractVector{T}, data=mld.data; verbose=false) where T
    return _marginalize(mld, v, data, mld.method, verbose)
end

"""Return the full dimension of the marginalized function, i.e. `length(u)` """
dimension(mld::MarginalLogDensity) = length(mld.u)

"""Return the indices of the marginalized variables, `iw`, in `u` """
imarginal(mld::MarginalLogDensity) = mld.iw

"""Return the indices of the non-marginalized variables, `iv`, in `u` """
ijoint(mld::MarginalLogDensity) = mld.iv

"""Return the number of marginalized variables."""
nmarginal(mld::MarginalLogDensity) = length(mld.u[mld.iw])

"""Return the number of non-marginalized variables."""
njoint(mld::MarginalLogDensity) = length(mld.u[mld.iv])

"""Get the value of the cached Hessian matrix."""
cached_hessian(mld::MarginalLogDensity) = mld.H

"""
Splice together the estimated (fixed) parameters `v` and marginalized (random) parameters
`w` into the single parameter vector `u`, based on their indices `iv` and `iw`.
"""
function merge_parameters(v::AbstractVector{T1}, w::AbstractVector{T2}, iv, iw) where {T1,T2}
    N = length(v) + length(w)
    u = Vector{promote_type(T1, T2)}(undef, N)
    u[iv] .= v
    u[iw] .= w
    return u
end

function ChainRulesCore.rrule(::typeof(merge_parameters), 
        v::AbstractVector{T1}, w::AbstractVector{T2}, iv, iw) where {T1,T2}
    u = merge_parameters(v, w, iv, iw)
    function merge_parameters_pullback(ubar)
        vbar = ubar[iv]
        wbar = ubar[iw]
        return (NoTangent(), vbar, wbar, NoTangent(), NoTangent())
    end
    return u, merge_parameters_pullback
end


"""
Split the vector of all parameters `u` into its estimated (fixed) components `v` and
marginalized (random) components `w`, based on their indices `iv` and `iw`.
components
"""
split_parameters(u, iv, iw) = (u[iv], u[iw])

function optimize_marginal!(mld, p2)
    w0 = mld.u[mld.iw]
    reinit!(mld.cache, u0=w0, p=p2)
    sol = solve!(mld.cache)
    wopt = sol.u::typeof(w0)
    objective = sol.objective::eltype(w0)
    mld.u[mld.iw] .= wopt
    return wopt, objective
end

function modal_hessian!(mld::MarginalLogDensity, w, p2)
    hessian!(mld.f_opt, mld.H, mld.hess_prep, mld.hess_adtype, w, Constant(p2))
    return mld.H
end

function _marginalize(mld, v, data, method::LaplaceApprox, verbose)
    p2 = (; p=data, v)
    verbose && println("Finding mode...")
    wopt, objective = optimize_marginal!(mld, p2)
    verbose && println("Calculating hessian...")
    modal_hessian!(mld, wopt, p2)
    verbose && println("Integrating...")
    nw = length(mld.iw)
    integral = -objective + (0.5nw) * log(2π) - 0.5logabsdet(mld.H)[1]
    verbose && println("Done!")
    return integral#, sol 
end

function hessdiag(f, x::Vector{T}) where T
    Δx = sqrt(eps(T))
    x .+= Δx
    g1 = ForwardDiff.gradient(f, x)
    x .-= 2Δx
    g2 = ForwardDiff.gradient(f, x)
    x .+= Δx # reset u
    return (g1 .- g2) ./ 2Δx
end

function _marginalize(mld, v, data, method::Cubature, verbose)
    p2 = (; p=data, v)
    if method.lower == nothing || method.upper == nothing
        wopt, _ = optimize_marginal!(mld, p2)
        println(wopt)
        h = hessdiag(w -> mld.f_opt(w, p2), wopt)
        se = 1 ./ sqrt.(h)
        upper = wopt .+ method.nσ * se
        lower = wopt .- method.nσ * se
    else
        lower = method.lower
        upper = method.upper
    end
    if verbose
        println(upper)
        println(lower)
    end
    integral, err = hcubature(w -> exp(-mld.f_opt(w, p2)), lower, upper)
    return log(integral)
end


function Optimization.OptimizationFunction(mld::MarginalLogDensity,
        args...; kwargs...)
    return OptimizationFunction((w, p) -> -mld(w, p), args...; kwargs...)
end

function Optimization.OptimizationProblem(mld::MarginalLogDensity, v0, p=mld.data;
    kwargs...)
    f = OptimizationFunction(mld)
    return OptimizationProblem(f, v0, p)
end

end # module
