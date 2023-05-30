module MarginalLogDensities
using ForwardDiff, FiniteDiff, ReverseDiff, Zygote
using Optimization
using OptimizationOptimJL
using LinearAlgebra
using SuiteSparse
using SparseArrays
using ChainRulesCore
using HCubature
using Distributions
using SparsityDetection
# using Symbolics
using SparseDiffTools
using NLSolversBase


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
    `LaplaceApprox([solver=LGFGS() [; adtype=Optimization.AutoForwardDiff(), opt_func_kwargs...]])

Construct a `LaplaceApprox` marginalizer to integrate out marginal variables via
the Laplace approximation. 
"""
struct LaplaceApprox{TA, TT, TS} <: AbstractMarginalizer
    # sparsehess::Bool
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
end

function LaplaceApprox(solver=LBFGS(); adtype=Optimization.AutoForwardDiff(),
    opt_func_kwargs...)
    return LaplaceApprox(solver, adtype, opt_func_kwargs)
end

struct Cubature{TA, TT, TS, T} <: AbstractMarginalizer
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
    upper::T
    lower::T
end

function Cubature(; solver=LBFGS(), adtype=Optimization.AutoForwardDiff(), 
    upper=nothing, lower=nothing, opt_func_kwargs...)
    return Cubature(solver, adtype, opt_func_kwargs, promote(upper, lower)...)
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
struct MarginalLogDensity{TF, TU<:AbstractVector, TD, TV<:AbstractVector, TW<:AbstractVector, 
        TF1<:OptimizationFunction, TM<:AbstractMarginalizer}
    logdensity::TF
    u::TU
    data::TD
    iv::TV
    iw::TW
    F::TF1
    method::TM
end

function get_hessian_prototype(f, w, p2, autosparsity)
    if autosparsity == :finitediff 
        H = FiniteDiff.finite_difference_hessian(w -> f(w, p2), w)
        hess_prototype = sparse(H) 
    elseif autosparsity == :forwarddiff
        H = ForwardDiff.hessian(w -> f(w, p2), w)
        hess_prototype = sparse(H)
    elseif autosparsity == :sparsitydetection
        hess_prototype = SparsityDetection.hessian_sparsity(w -> f(w, p2), w) .* one(eltype(w))
    # elseif autosparsity == :symbolics
    #     ...
    elseif autosparsity == :none
        hess_prototype = ones(eltype(w), length(w), length(w))
    else
        error("Unsupported method for hessian sparsity detection: $(autosparsity)")
    end
    return hess_prototype
end

# hess_autosparse = :none, :finitediff :forwarddiff, :sparsitydetection, (:symbolics)
function MarginalLogDensity(logdensity, u, iw, data=(), method=LaplaceApprox(); hess_autosparse=:none)
    n = length(u)
    iv = setdiff(1:n, iw)
    w = u[iw]
    v = u[iv]
    p2 = (p=data, v=v)
    f(w, p2) = -logdensity(merge_parameters(p2.v, w, iv, iw), p2.p)
    hess_prototype = get_hessian_prototype(f, w, p2, hess_autosparse)
    if hess_autosparse != :none
        hess_colorvec = matrix_colors(hess_prototype)
        hess = (H, w, p2) -> numauto_color_hessian!(H, w -> f(w, p2), w, hess_colorvec, hess_prototype)
        F = OptimizationFunction(f, method.adtype; hess_prototype=hess_prototype, hess_colorvec=hess_colorvec,
            hess = hess, method.opt_func_kwargs...)
    else
        hess = (H, w, p2) -> ForwardDiff.hessian!(H, w -> f(w, p2), w) #auto_ad_hess(method.adtype)(H, u, p)
        F = OptimizationFunction(f, method.adtype; hess_prototype=hess_prototype,
            hess = hess, method.opt_func_kwargs...)
    end
    return MarginalLogDensity(logdensity, u, data, iv, iw, F, method)
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
        return NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
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
    prob = OptimizationProblem(mld.F, w0, p2)
    sol = solve(prob, mld.method.solver)
    wopt = sol.u
    mld.u[mld.iw] = wopt
    return sol
end

function _marginalize(mld, v, data, method::LaplaceApprox, verbose)
    p2 = (; p=data, v)
    verbose && println("Finding mode...")
    sol = optimize_marginal!(mld, p2)
    verbose && println("Calculating hessian...")
    # H = -ForwardDiff.hessian(w -> mld.F(w, p2), sol.u)
    H = mld.F.hess(mld.F.hess_prototype, sol.u, p2)
    verbose && println("Integrating...")
    nw = length(mld.iw)
    integral = -sol.objective + (nw/2)* log(2π) - 0.5logabsdet(H)[1]
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
        sol = optimize_marginal!(mld, p2)
        wopt = sol.u
        h = hessdiag(w -> mld.F(w, p2), wopt)
        se = 1 ./ sqrt.(h)
        upper = wopt .+ 6se
        lower = wopt .- 6se
    else
        lower = method.lower
        upper = method.upper
    end
    println(upper)
    println(lower)
    integral, err = hcubature(w -> exp(-mld.F(w, p2)), lower, upper)
    return log(integral)
end

function Optim.optimize(mld::MarginalLogDensity, init_v, data=(), args...; kwargs...)
    return optimize(v -> -mld(v, data), init_v, args...; kwargs...)
end

end