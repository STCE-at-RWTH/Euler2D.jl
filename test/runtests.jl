using Test
using Zygote

"""
Jacobian wrt. `u` of the flux function.

Outputs a 2 x n x 2 where n is the number of space dims
"""
function ∇F(u; gas::CaloricallyPerfectGas = DRY_AIR)
    n_u = length(u)
    n_x = n_u - 2
    _, F_back = pullback(u) do u
        F(u; gas = gas)
    end
    seeds = [begin
        b = zeros((n_u, n_x))
        b[i, j] = 1.0
        b
    end for j = 1:n_x for i = 1:n_u]
    ∂F = map(seeds) do F̄
        F_back(F̄)[1]
    end
    output_ranges = [range(1 + i * n_u; length = n_u) for i = 0:(n_x-1)]
    out = stack(map(output_ranges) do r
        reduce(hcat, ∂F[r])'
    end)
    return permutedims(out, (1, 3, 2))
end
