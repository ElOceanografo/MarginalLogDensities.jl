module MarginalLogDensities
using ForwardDiff, FiniteDiff, ReverseDiff, Zygote
using Optimization
using OptimizationOptimJL
using LinearAlgebra
using SuiteSparse
using SparseArrays
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

struct LaplaceApprox{TA, TT, TS} <: AbstractMarginalizer
    # sparsehess::Bool
    solver::TS
    adtype::TA
    opt_func_kwargs::TT
end

# function LaplaceApprox(sparsehess=false, adtype=AutoForwardDiff(), opt_func_kwargs=(;))
function LaplaceApprox(solver=LBFGS(); adtype=Optimization.AutoForwardDiff(),
    opt_func_kwargs...)
    return LaplaceApprox(solver, adtype, opt_func_kwargs)
end

#=
in MarginalLogDensity constructor, can then do:
    if sparse
    # Hcolors, Hsparsity = get_hessian_sparsity(logdensity, u, forwarddiff_sparsity)
    f(w, p2) = -logdensity(merge_parameters(p2.v, w, iv, iw), p2.p)
    F = OptimizationFunction(f, method.adtype; method.kwargs...)
=#

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
    `MarginalLogDensity(logdensity, n, imarginal, [method=LaplaceApprox()])`

Construct a callable object which wraps the function `logdensity` and
integrates over a subset of its arguments.
* `logdensity` : function with signature `(u, p)` returning a positive
log-probability (e.g. a log-pdf, log-likelihood, or log-posterior).
* `imarginal` : Vector of indices indicating which arguments to `logdensity` to marginalize
* `method` : How to perform the marginalization.  Defaults to `LaplaceApprox()`; `Cubature()`
is also available.

If `length(imarginal) == m`, then the constructed `MarginalLogDensity` object  `mld`
can be called as `mld(θ)`, where `θ` is a vector with length `n-m`.  It can also be called
as `mld(u, θ)`, where `u` is a length-`m` vector of the marginalized variables.  In this
case, the return value is the same as the full conditional `logdensity` with `u` and `θ`
"""
struct MarginalLogDensity{TF, TU<:AbstractVector, TP, TV<:AbstractVector, TW<:AbstractVector, 
        TF1<:OptimizationFunction, TM<:AbstractMarginalizer}
    logdensity::TF
    u::TU
    p::TP
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

# autosparsity = :none, :finitediff :forwarddiff, :sparsitydetection, :symbolics
function MarginalLogDensity(logdensity, u, iw, p=(), method=LaplaceApprox(); hess_autosparse=:none)
    n = length(u)
    iv = setdiff(1:n, iw)
    w = u[iw]
    v = u[iv]
    p2 = (p=p, v=v)
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
    return MarginalLogDensity(logdensity, u, p, iv, iw, F, method)
end

function Base.show(io::IO, mld::MarginalLogDensity)
    T = typeof(mld.method).name.name
    str = "MarginalLogDensity of function $(repr(mld.logdensity))\nIntegrating $(length(mld.iw))/$(length(mld.u)) variables via $(T)"
    write(io, str)
end

function (mld::MarginalLogDensity)(v::AbstractVector{T}, p; verbose=false) where T
    return _marginalize(mld, v, p, mld.method, verbose)
end

dimension(mld::MarginalLogDensity) = length(mld.u)
imarginal(mld::MarginalLogDensity) = mld.iw
ijoint(mld::MarginalLogDensity) = mld.iv
nmarginal(mld::MarginalLogDensity) = length(mld.iw)
njoint(mld::MarginalLogDensity) = length(mld.iv)
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
# TODO: add ChainRules for merge_parameters so it works w/ Zygote


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

function _marginalize(mld, v, p, method::LaplaceApprox, verbose)
    p2 = (;p, v)
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

function _marginalize(mld, v, p, method::Cubature, verbose)
    p2 = (;p, v)
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

end

# Was using this for testing/troubleshooting, will probably delete later
# """
#     `num_hessian_sparsity(f, x, [δ=1.0])`
#
# Calculate the sparsity pattern of the Hessian matrix of function `f`. This is a brute-force
# approach, but more robust than the one in SparsityDetection
# """
# function num_hessian_sparsity(f, x, δ=1.0)
#     N = length(x)
#     g(x) = ForwardDiff.gradient(f, x)
#     y = g(x)
#     ii = Int[]
#     jj = Int[]
#     vv = Float64[]
#     for j in 1:N
#         x[j] += δ
#         yj = g(x)
#         di = findall(.! (yj .≈ y))
#         for i in di
#             push!(jj, j)
#             push!(ii, i)
#             push!(vv, 1.0)
#         end
#         x[j] -= δ
#     end
#     return sparse(ii, jj, vv)
# end


