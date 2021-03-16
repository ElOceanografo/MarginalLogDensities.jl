module LaplaceApproximation

using Optim
using ForwardDiff
using LinearAlgebra
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
struct Cubature <: AbstractMarginalizer end


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
    objective = θmarginal -> -mld(θmarginal, θjoint)
    opt = optimize(objective, ones(nmarginal(mld)))
    H = ForwardDiff.hessian(objective, opt.minimizer)
    logz = -opt.minimum + 0.5 * log(2π) * logdet(H)
    return logz
end


end # module
