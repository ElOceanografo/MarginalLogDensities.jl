module MarginalLogDensities
using Optimization
using OptimizationOptimJL
using ForwardDiff, FiniteDiff, ReverseDiff, Zygote
using DifferentiationInterface
using ADTypes
using LinearAlgebra
using SparseArrays
using ChainRulesCore
using HCubature
using Distributions

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
    optimize_marginal!,
    hessdiag,
    get_hessian_sparsity

# can't seem to precompile these functions
# auto_ad_hess(x::Optimization.AutoFiniteDiff) = FiniteDiff.finite_difference_hessian!
# auto_ad_hess(x::Optimization.AutoForwardDiff) = ForwardDiff.hessian!
# auto_ad_hess(x::Optimization.AutoReverseDiff) = ReverseDiff.hessian!
# auto_ad_hess(x::Optimization.AutoZygote) = (H, f, x) -> first(Zygote.hessian!(H, f, x))

abstract type AbstractMarginalizer end

"""
    `LaplaceApprox([solver=LBFGS() [; adtype=Optimization.AutoForwardDiff(), opt_func_kwargs...]])

Construct a `LaplaceApprox` marginalizer to integrate out marginal variables via
the Laplace approximation. This method will usually be faster than `Cubature`, especially
in high dimensions, though it may not be as accurate.

# Arguments
- `solver=LBFGS()` : Algorithm to use when performing the inner optimization to find the
mode of the marginalized variables. Can be any algorithm defined in Optim.jl.
- `adtype=Optimization.AutoForwardDiff()` : Automatic differentiation type to use for the 
inner optimization. `AutoForwardDiff()` is robust and fast for small problems; for larger
ones `AutoReverseDiff()` or `AutoZygote()` are likely better.
- `opt_func_kwargs` : Optional keyword arguments passed on to `Optimization.OptimizationFunction`.
"""
struct LaplaceApprox{TA, TT, TS} <: AbstractMarginalizer
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
end

function LaplaceApprox(solver=LBFGS(); adtype=Optimization.AutoForwardDiff(),
    opt_func_kwargs...)
    return LaplaceApprox(solver, adtype, opt_func_kwargs)
end

"""
    Cubature([; solver=LBFGS(), adtype=Optimization.AutoForwardDiff(),
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
- `adtype=Optimization.AutoForwardDiff()` : Automatic differentiation type to use for the 
inner optimization. `AutoForwardDiff()` is robust and fast for small problems; for larger
ones `AutoReverseDiff()` or `AutoZygote()` are likely better.
- `upper`, `lower` : Optional upper and lower bounds for the numerical integration. If supplied,
they must be numeric vectors the same length as the marginal variables.
- `nσ=6.0` : If `upper` and `lower` are not supplied, integrate this many standard deviations
away from the mode based on a Laplace approximation to the curvature there.
- `opt_func_kwargs` : Optional keyword arguments passed on to `Optimization.OptimizationFunction`.
"""
struct Cubature{TA, TT, TS, TV, T} <: AbstractMarginalizer
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
    upper::TV
    lower::TV
    nσ::T
end

function Cubature(; solver=LBFGS(), adtype=Optimization.AutoForwardDiff(), 
    upper=nothing, lower=nothing, nσ=6.0, opt_func_kwargs...)
    return Cubature(solver, adtype, opt_func_kwargs, promote(upper, lower)..., nσ)
end

function get_hessian_prototype(f, w, p2, autosparsity)
    f2(w) = f(w, p2)
    if autosparsity == :finitediff 
        H = FiniteDiff.finite_difference_hessian(f2, w)
        hess_prototype = sparse(H) 
    elseif autosparsity == :forwarddiff
        H = ForwardDiff.hessian(f2, w)
        hess_prototype = sparse(H)
    elseif autosparsity == :reversediff
        H = ReverseDiff.hessian(f2, w)
        hess_prototype = sparse(H)
    elseif autosparsity == :zygote
        H = Zygote.hessian(f2, w)
        hess_prototype = sparse(H)
    # elseif autosparsity == :sparsitydetection
        # hess_prototype = SparsityDetection.hessian_sparsity(w -> f(w, p2), w) .* one(eltype(w))
    # elseif autosparsity == :symbolics
    #     ...
    elseif autosparsity == :none
        hess_prototype = ones(eltype(w), length(w), length(w))
    else
        error("Unsupported method for hessian sparsity detection: $(autosparsity)")
    end
    return hess_prototype
end


"""
    `MarginalLogDensity(logdensity, u, iw, data, [method=LaplaceApprox()])`

Construct a callable object which wraps the function `logdensity` and
integrates over a subset of its arguments.

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
- `hess_autosparse=:none` : Specifies how to detect sparsity in the Hessian matrix of 
`logdensity`. Can be `:none`, `:finitediff`` `:forwarddiff`, or `:sparsitydetection`.
If `:none` (the default), the Hessian is assumed dense and calculated using `ForwardDiff`. 
Detecting sparsity takes some time and may not be worth it for small problems, but for 
larger problems it can be extremely worth it.

The resulting `MarginalLogDensity` object  `mld` can then be called like a function
as `mld(v, data)`, where `v` is the subset of the full parameter vector `u` which is
*not* indexed by `iw`.  If `length(u) == n` and `length(iw) == m`, then `length(v) == n-m`.

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
        TE<:DifferentiationInterface.HessianExtras
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
    hess_extras::TE
end


function MarginalLogDensity(logdensity, u, iw, data=(), method=LaplaceApprox(); 
        hess_autosparse=:none,
        hess_adtype=SecondOrder(AutoSparseForwardDiff(), AutoReverseDiff()))
    n = length(u)
    iv = setdiff(1:n, iw)
    w = u[iw]
    v = u[iv]
    p2 = (p=data, v=v)
    f(w, p2) = -logdensity(merge_parameters(p2.v, w, iv, iw), p2.p)
    H = get_hessian_prototype(f, w, p2, hess_autosparse)
    f_opt = OptimizationFunction(f, method.adtype; hess_prototype=H,
        method.opt_func_kwargs...)
    prob = OptimizationProblem(f_opt, w, p2)
    cache = init(prob, method.solver) 
    extras = prepare_hessian(w -> f(w, p2), hess_adtype, w)
    return MarginalLogDensity(logdensity, u, data, iv, iw, method, f_opt, prob, cache, H,
        hess_adtype, extras)
end

function Base.show(io::IO, mld::MarginalLogDensity)
    T = typeof(mld.method).name.name
    str = "MarginalLogDensity of function $(repr(mld.logdensity))\nIntegrating $(length(mld.iw))/$(length(mld.u)) variables via $(T)"
    write(io, str)
end

function (mld::MarginalLogDensity)(v::AbstractVector{T}, data; verbose=false) where T
    return _marginalize(mld, v, data, mld.method, verbose)
end

"""Return the full dimension of the marginalized function, i.e. `length(u)` """
dimension(mld::MarginalLogDensity) = length(mld.u)

"""Return the indices of the marginalized variables, `iw`, in `u` """
imarginal(mld::MarginalLogDensity) = mld.iw

"""Return the indices of the non-marginalized variables, `iv`, in `u` """
ijoint(mld::MarginalLogDensity) = mld.iv

"""Return the number of marginalized variables."""
nmarginal(mld::MarginalLogDensity) = length(mld.iw)

"""Return the number of non-marginalized variables."""
njoint(mld::MarginalLogDensity) = length(mld.iv)

"""Get the value of the cached Hessian matrix."""
cached_hessian(mld::MarginalLogDensity) = mld.F.hess_prototype

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
    hessian!(w -> mld.f_opt(w, p2), mld.H, mld.hess_adtype, w, mld.hess_extras)
    return mld.H
end

function _marginalize(mld, v, data, method::LaplaceApprox, verbose)
    p2 = (; p=data, v)
    verbose && println("Finding mode...")
    wopt, objective = optimize_marginal!(mld, p2)
    verbose && println("Calculating hessian...")
    modal_hessian!(mld, wopt, p2)
    # H = Diagonal(hessdiag(w -> mld.f_opt(w, p2), wopt))
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

# function Optim.optimize(mld::MarginalLogDensity, init_v, data=(), args...; kwargs...)
#     return optimize(v -> -mld(v, data), init_v, args...; kwargs...)
# end

end # module
