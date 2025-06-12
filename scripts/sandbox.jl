using BenchmarkTools
using Euler2D
using Euler2D: ϕ_hll
using ForwardDiff
using ForwardDiff: JacobianConfig
using LinearAlgebra
using ShockwaveProperties
using StaticArrays
using Unitful

state_left = PrimitiveProps(1.0u"kg/m^3", SVector(0.0, 0.0), 10.0u"kPa", DRY_AIR)
state_right = PrimitiveProps(0.125u"kg/m^3", SVector(0.0, 0.0), 1.0u"kPa", DRY_AIR)

u_left = state_to_vector(ConservedProps(state_left, DRY_AIR))
u_right = state_to_vector(ConservedProps(state_right, DRY_AIR))
u_arg = vcat(u_left, u_right)

u̇_left = SVector(1.0, 0.0, 0.0, 0.0)
u̇_right = SVector(0.0, 0.0, 0.0, 0.0)

ϕ_hll(u_left, u_right, 1, DRY_AIR)

function split_svector(v)
    N = length(v) ÷ 2
    M = length(v)
    v1, v2 = @inbounds begin
        (SVector{N}(@view v[1:N]), SVector{M-N}(@view v[N+1:M]))
    end
    return v1,v2
end

function ϕ_hll_fused_args(u_arg, dim, gas::CaloricallyPerfectGas)
    uL, uR = split_svector(u_arg)
    return ϕ_hll(uL, uR, dim, gas)
end

function ϕ_hll_jvp(uL, u̇L, uR, u̇R, dim, gas::CaloricallyPerfectGas)
    u_arg = vcat(uL, uR)
    J = ForwardDiff.jacobian(u->ϕ_hll_fused_args(u,dim,gas), u_arg)
    return J * vcat(u̇L, u̇R)
end

ϕ_hll_jvp(u_left, u̇_left, u_right, u̇_right, 2, DRY_AIR)

@benchmark ϕ_hll_jvp($u_left, $u̇_left, $u_right, $u̇_right, 1, DRY_AIR)