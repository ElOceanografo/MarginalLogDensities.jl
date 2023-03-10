module MarginalLogDensities
using Optimization
using OptimizationOptimJL
using ForwardDiff
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
    merge_parameters,
    split_parameters

abstract type AbstractMarginalizer end

# struct LaplaceApprox <: AbstractMarginalizer end
## struct LaplaceApprox <: AbstractMarginalizer
struct LaplaceApprox{TA, TT, TS} <: AbstractMarginalizer
    # sparsehess::Bool
    adtype::TA
    opt_func_kwargs::TT
    solver::TS
end

# function LaplaceApprox(sparsehess=false, adtype=AutoForwardDiff(), opt_func_kwargs=(;))
function LaplaceApprox(adtype=Optimization.AutoForwardDiff(), opt_func_kwargs=(;); 
        solver=LBFGS())
   return LaplaceApprox(adtype, opt_func_kwargs, solver)
end

#=
in MarginalLogDensity constructor, can then do:
    if sparse
    # Hcolors, Hsparsity = get_hessian_sparsity(logdensity, u, forwarddiff_sparsity)
    f(w, p2) = -logdensity(merge_parameters(p2.v, w, iv, iw), p2.p)
    F = OptimizationFunction(f, method.adtype; method.kwargs...)
=#

struct Cubature{TA, TT, T} <: AbstractMarginalizer
    adtype::TA
    opt_func_kwargs::TT
    upper::AbstractVector{T}
    lower::AbstractVector{T}
end
function Cubature(upper::T1, lower::T2) where {T1, T2}
    return Cubature(Optimization.AutoForwardDiff(), (;), promote(upper, lower)...)
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
struct MarginalLogDensity{TF, TU<:AbstractVector, TV<:AbstractVector, TW<:AbstractVector, 
        TF1<:OptimizationFunction, TM<:AbstractMarginalizer}
    logdensity::TF
    u::TU
    iv::TV
    iw::TW
    F::TF1
    method::TM
end

function get_hessian_sparsity(f, u, forwarddiff_sparsity)
    if forwarddiff_sparsity
        println("Detecting Hessian sparsity via ForwardDiff...")
        H = ForwardDiff.hessian(f, u)
        Hsparsity = sparse(H) .!= 0
    else
        println("Detecting Hessian sparsity via SparsityDetection...")
        Hsparsity = hessian_sparsity(f, u)
    end
    Hcolors = matrix_colors(Hsparsity)

    return Hsparsity, Hcolors
end

function MarginalLogDensity(logdensity, u, iw, method=LaplaceApprox(), forwarddiff_sparsity=false)
    n = length(u)
    iv = setdiff(1:n, iw)
    # hess_sparsity, hess_colors, hess = get_hessian_sparsity(f, u, iw, forwarddiff_sparsity)
    f(w, p2) = -logdensity(merge_parameters(p2.v, w, iv, iw), p2.p)
    F = OptimizationFunction(f, method.adtype; method.opt_func_kwargs...)
    return MarginalLogDensity(logdensity, u, iv, iw, F, method)
end

function Base.show(io::IO, mld::MarginalLogDensity)
    str = "MarginalLogDensity of function $(repr(mld.logdensity))\nIntegrating $(length(mld.iw))/$(length(mld.u)) variables via $(repr(mld.method))"
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

function _marginalize(mld, v, p, method::LaplaceApprox, verbose)
    w0 = mld.u[mld.iw]
    nw = length(w0)
    p2 = (;p, v)
    verbose && println("Finding mode...")
    prob = OptimizationProblem(mld.F, w0, p2)
    sol = solve(prob, method.solver)
    wopt = sol.u
    verbose && println("Calculating hessian...")
    H = -ForwardDiff.hessian(w -> mld.F(w, p2), wopt)
    mld.u[mld.iw] = wopt
    verbose && println("Integrating...")
    integral = -sol.objective + (nw/2)* log(2π) - 0.5logabsdet(H)[1]
    verbose && println("Done!")
    return integral#, sol 
end

function _marginalize(mld, v, p, method::Cubature, verbose)
    p2 = (;p, v)
    integral, err = hcubature(w -> exp(-mld.F(w, p2)), method.lower, method.upper)
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


