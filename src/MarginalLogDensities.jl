module MarginalLogDensities

using Optim
using ForwardDiff
using LinearAlgebra
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
    njoint

abstract type AbstractMarginalizer end

struct LaplaceApprox <: AbstractMarginalizer end

struct Cubature{T} <: AbstractMarginalizer
    upper::AbstractVector{T}
    lower::AbstractVector{T}
end


struct MarginalLogDensity{TI<:Integer, TM<:AbstractMarginalizer,
        TV<:AbstractVector{TI}, TF}
    logdensity::TF
    dimension::TI
    imarginal::TV
    ijoint::TV
    method::TM
end

function MarginalLogDensity(logdensity::Function, dimension::TI,
        imarginal::AbstractVector{TI}, method=LaplaceApprox()) where {TI<:Integer}
    ijoint = setdiff(1:dimension, imarginal)
    return MarginalLogDensity(logdensity, dimension, imarginal, ijoint, method)
end

dimension(mld::MarginalLogDensity) = mld.dimension
imarginal(mld::MarginalLogDensity) = mld.imarginal
ijoint(mld::MarginalLogDensity) = mld.ijoint
nmarginal(mld::MarginalLogDensity) = length(mld.imarginal)
njoint(mld::MarginalLogDensity) = length(mld.ijoint)


function (mld::MarginalLogDensity)(θmarg::AbstractVector{T1}, θjoint::AbstractVector{T2}) where {T1, T2}
    θ = Vector{promote_type(T1, T2)}(undef, dimension(mld))
    θ[imarginal(mld)] .= θmarg
    θ[ijoint(mld)] .= θjoint
    return mld.logdensity(θ)
end

function (mld::MarginalLogDensity)(θjoint::AbstractVector{T}) where T
    logz = _marginalize(mld, θjoint, mld.method)
    return logz
end

function _marginalize(mld::MarginalLogDensity, θjoint::AbstractVector{T},
        method::LaplaceApprox) where T
    f = θmarginal -> -mld(θmarginal, θjoint)
    N = nmarginal(mld)
    opt = optimize(f, zeros(N))
    H = -ForwardDiff.hessian(f, opt.minimizer)
    logz = -opt.minimum + 0.5 * (log((2π)^N) - logdet(H))
end

function _marginalize(mld::MarginalLogDensity, θjoint, method::Cubature)
    f = θmarginal -> exp(mld(θmarginal, θjoint))
    int, err = hcubature(f, method.lower, method.upper)
    return log(int)
end

end # module
