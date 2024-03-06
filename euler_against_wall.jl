using ShockwaveProperties
using ShockwaveProperties.BilligShockParametrization
using Zygote

##

function F(u; gas::CaloricallyPerfectGas=DRY_AIR)
    ρv = @view u[2:end-1]
    v = ρv / u[1]
    P = ustrip(pressure(internal_energy_density(u[1], ρv, u[end]); gas=gas))
    return vcat(ρv', ρv .* v' + I * P, (v .* (u[end] + P))')
end

F_n(u, n̂; gas::CaloricallyPerfectGas=DRY_AIR) = F(u) * n̂

function ∂F∂u(u; gas::CaloricallyPerfectGas=DRY_AIR)
    n = length(u)
    _, F_back = pullback(u -> F(u; gas=gas), u)
    seeds = vcat([hcat(1.0 * I[1:n, k], zeros(n)) for k ∈ 1:n],
        [hcat(zeros(n), 1.0 * I[1:n, k]) for k ∈ 1:n])
    ∂F = map(seeds) do F̄
        F_back(F̄)[1]
    end
    out = stack([reduce(hcat, ∂F[1:n])', reduce(hcat, ∂F[(n+1):end])'])
    return permutedims(out, (1, 3, 2))
end

##

function step_euler(u_grid; gas::CaloricallyPerfectGas=DRY_AIR)

end