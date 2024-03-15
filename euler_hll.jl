using LinearAlgebra
using Tullio
using ShockwaveProperties
using Unitful
using Zygote

##

free_stream = [0.8, 2.0, 0.0, 100]
uinf = conserved_state_vector(free_stream; gas=DRY_AIR)
u_grid = stack(reshape([uinf for i = 1:100 for j = 1:100], (100, 100)))
ws = stack(map(u -> eigenvalues_∇F(u; gas=DRY_AIR), eachslice(u_grid; dims=(2, 3))))
F_grid = stack(map(u -> F(u; gas=DRY_AIR), eachslice(u_grid; dims=(2, 3))))

##

function between(ival, x::T) where {T}
    proper = (ival[1] < ival[2])
    if proper && (ival[1] < x < ival[2])
        return one(T)
    elseif !proper && (ival[2] < x < ival[1])
        return -one(T)
    else
        return zero(T)
    end
end
@inline X_in(ival) = Base.Fix1(between, ival)

function pressure_u(u; gas::CaloricallyPerfectGas=DRY_AIR)
    ρe = internal_energy_density(u[1], u[2:end-1], u[end])
    return (gas.γ - 1) * ρe
end

function speed_of_sound_u(u; gas::CaloricallyPerfectGas=DRY_AIR)
    P = pressure_u(u; gas=gas)
    return sqrt(gas.γ * P / u[1])
end

function F(u; gas::CaloricallyPerfectGas=DRY_AIR)
    ρv = @view u[2:end-1]
    v = ρv / u[1]
    P = pressure_u(u; gas=gas)
    return vcat(ρv', ρv .* v' + I * P, (v .* (u[end] + P))')
end

F_n(u, n̂; gas::CaloricallyPerfectGas=DRY_AIR) = F(u; gas=gas) * n̂

"""
    Jacobian wrt. u of the flux function.
    outputs a 2 x n x 2 where n is the number of space dims
"""
function ∇F(u; gas::CaloricallyPerfectGas=DRY_AIR)
    n_u = length(u)
    n_x = n_u - 2
    _, F_back = pullback(u) do u
        F(u; gas=gas)
    end
    seeds = [
        begin
            b = zeros((n_u, n_x))
            b[i, j] = 1.0
            b
        end for j = 1:n_x for i = 1:n_u
    ]
    ∂F = map(seeds) do F̄
        F_back(F̄)[1]
    end
    output_ranges = [range(1 + i * n_u; length=n_u) for i = 0:(n_x-1)]
    out = stack(map(output_ranges) do r
        reduce(hcat, ∂F[r])'
    end)
    return permutedims(out, (1, 3, 2))
end

# do we need the multiple eigenvalues in the middle? I do not know...
function eigenvalues_∇F(u; gas::CaloricallyPerfectGas=DRY_AIR)
    ndims = length(u)
    a = speed_of_sound_u(u; gas=gas)
    out = repeat((u[2:end-1] ./ u[1])', ndims, 1)
    out[1, :] .-= a
    out[end, :] .+= a
    return out
end

"""
Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `FL`, `FR`: Flux in the appropriate direction on either side of the boundary
- `aL`, `aR`: Wave speeds in the appropriate direction on either side of the boundary
"""
function hll_numerical_flux(uL, uR, FL, FR, aL, aR)
    sL = min(minimum(aL), minimum(aR))
    sR = max(maximum(aL), maximum(aR))
    if sL >= 0
        return FL
    elseif sR <= 0
        return FR
    else
        return (sR * FL - sL * FR + sL * sR * (uL - uR)) / (sR - sL)
    end
end

"""
Compute the HLL numerical flux without applying boundary conditions.

Returns F_hll[:, :, i, j] where we identify F_hll[:, :, i, j] with F̂_{i+1/2, j+1/2}
"""
function numerical_flux_hll(u, wave_speeds; gas::CaloricallyPerfectGas=DRY_AIR)
    cell_flux = stack(map(u -> F(u; gas=gas), eachslice(u; dims=(2, 3))))
    @tullio F_hll[:, 1, i, j] = choose_hll_flux(
        u[:, i, j], u[:, i+1, j],
        cell_flux[:, 1, i, j], cell_flux[:, 1, i+1, j],
        min(minimum(wave_speeds[:, 1, i, j]), minimum(wave_speeds[:, 1, i+1, j])),
        max(maximum(wave_speeds[:, 1, i, j]), maximum(wave_speeds[:, 1, i+1, j]))
    )
    @tullio F_hll[:, 2, i, j] = choose_hll_flux(
        u[:, i, j], u[:, i, j+1],
        cell_flux[:, 2, i, j], cell_flux[:, 2, i, j+1],
        min(minimum(wave_speeds[:, 2, i, j]), minimum(wave_speeds[:, 2, i, j+1])),
        max(maximum(wave_speeds[:, 2, i, j]), maximum(wave_speeds[:, 2, i, j+1]))
    )
    return F_hll
end

function enforce_reflection_boundary(u, F_num, wall_dim, axis)
    # we create an identical ghost cell on the other side of the wall
    # and say u_{-1} = u_{1}
    
end

function weakly_enforce_wall_boundary(u, F̂, wall_dim, axis)
    # extrapolate the pressure to the wall and
    # enforce that F̂ along the wall is equal to the pressure
end

function enforce_inflow_boundary()
    # how

end

function enforce_outflow_boundary()
    # HOW
end

##

"""
- `u` is `4` by `nx` by `ny`. i.e. `4x100x100` for a 10,000 grid cell solver in 2D.
"""
function step_euler_eqns_2d_hll(u, Δx; gas::CaloricallyPerfectGas=DRY_AIR, CFL=0.75)
    wave_speeds = stack(map(u -> eigenvalues_∇F(u; gas=gas), eachslice(u; dims=(2, 3))))
    a_max = max(abs(minimum(wave_speeds)), abs(maximum(wave_speeds)))
    Δt = CFL * Δx / a_max
    cell_flux = stack(map(u -> F(u; gas=gas), eachslice(u; dims=(2, 3))))
    @tullio F_hll[:, 1, i, j] = choose_hll_flux(
        u[:, i, j], u[:, i+1, j],
        cell_flux[:, 1, i, j], cell_flux[:, 1, i+1, j],
        min(minimum(wave_speeds[:, 1, i, j]), minimum(wave_speeds[:, 1, i+1, j])),
        max(maximum(wave_speeds[:, 1, i, j]), maximum(wave_speeds[:, 1, i+1, j]))
    )
    @tullio F_hll[:, 2, i, j] = choose_hll_flux(
        u[:, i, j], u[:, i, j+1],
        cell_flux[:, 2, i, j], cell_flux[:, 2, i, j+1],
        min(minimum(wave_speeds[:, 2, i, j]), minimum(wave_speeds[:, 2, i, j+1])),
        max(maximum(wave_speeds[:, 2, i, j]), maximum(wave_speeds[:, 2, i, j+1]))
    )
    return F_hll

end