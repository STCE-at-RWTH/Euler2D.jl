using LinearAlgebra
using Tullio
using ShockwaveProperties
using Unitful
using Zygote

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

function pressure_u(u; gas::CaloricallyPerfectGas = DRY_AIR)
    ρe = internal_energy_density(u[1], u[2:end-1], u[end])
    return (gas.γ - 1) * ρe
end

function speed_of_sound_u(u; gas::CaloricallyPerfectGas = DRY_AIR)
    P = pressure_u(u; gas = gas)
    return sqrt(gas.γ * P / u[1])
end

function F(u; gas::CaloricallyPerfectGas = DRY_AIR)
    ρv = @view u[2:end-1]
    v = ρv / u[1]
    P = pressure_u(u; gas = gas)
    return vcat(ρv', ρv .* v' + I * P, (v .* (u[end] + P))')
end

F_n(u, n̂; gas::CaloricallyPerfectGas = DRY_AIR) = F(u; gas = gas) * n̂

"""
    Jacobian wrt. u of the flux function.
    outputs a 2 x n x 2 where n is the number of space dims
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

# do we need the multiple eigenvalues in the middle? I do not know...
function eigenvalues_∇F(u; gas::CaloricallyPerfectGas = DRY_AIR)
    ndims = length(u)
    a = speed_of_sound_u(u; gas = gas)
    out = repeat((u[2:end-1] ./ u[1])', ndims, 1)
    out[1, :] .-= a
    out[end, :] .+= a
    return out
end

"""
Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `dim` : Direction to calculate F_hll
"""
function F_hll(uL, uR, dim; gas::CaloricallyPerfectGas = DRY_AIR)
    FL = F(uL; gas = gas)[:, dim]
    FR = F(uR; gas = gas)[:, dim]
    aL = eigenvalues_∇F(uL; gas = gas)[:, dim]
    aR = eigenvalues_∇F(uR; gas = gas)[:, dim]
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

_ghost_u(u, dim) = begin
    ghost = copy(u)
    ghost[dim+1] *= -1
    return ghost
end

function wall_boundaries!(u_next, u, Δx, Δy, Δt; gas::CaloricallyPerfectGas = DRY_AIR)
    # extrapolate the pressure to the wall and
    # enforce that F̂ along the wall is equal to the pressure
    # and that v⋅n is zero

    # corners
    u_next[:, 1, 1] =
        u[:, 1, 1] + (
            Δt / Δx * (
                F_hll(_ghost_u(u[:, 1, 1], 1), u[:, 1, 1], 1; gas = gas) -
                F_hll(u[:, 1, 1], u[:, 2, 1], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(_ghost_u(u[:, 1, 1], 2), u[:, 1, 1], 2; gas = gas) -
                F_hll(u[:, 1, 1], u[:, 1, 2], 2; gas = gas)
            )
        )
    u_next[:, 1, end] =
        u[:, 1, end] + (
            Δt / Δx * (
                F_hll(_ghost_u(u[:, 1, end], 1), u[:, 1, end], 1; gas = gas) -
                F_hll(u[:, 1, end], u[:, 2, end], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, 1, end-1], u[:, 1, end], 2; gas = gas) -
                F_hll(u[:, 1, end], _ghost_u(u[:, 1, end], 2), 2; gas = gas)
            )
        )
    u_next[:, end, 1] =
        u[:, end, 1] + (
            Δt / Δx * (
                F_hll(u[:, end-1, 1], u[:, end, 1], 1; gas = gas) -
                F_hll(u[:, end, 1], _ghost_u(u[:, end, 1], 1), 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(_ghost_u(u[:, end, 1], 2), u[:, end, 1], 2; gas = gas) -
                F_hll(u[:, end, 1], u[:, end, 2], 2; gas = gas)
            )
        )
    u_next[:, end, end] =
        u[:, end, end] + (
            Δt / Δx * (
                F_hll(u[:, end-1, end], u[:, end, end], 1; gas = gas) -
                F_hll(u[:, end, end], _ghost_u(u[:, end, end], 1), 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, end, end-1], u[:, end, end], 2; gas = gas) -
                F_hll(u[:, end, end], _ghost_u(u[:, end, end], 2), 2; gas = gas)
            )
        )

    # bottom wall
    @tullio u_next[:, i, 1] =
        u[:, i, 1] + (
            Δt / Δx * (
                F_hll(u[:, i-1, 1], u[:, i, 1], 1; gas = gas) -
                F_hll(u[:, i, 1], u[:, i+1, 1], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(_ghost_u(u[:, i, 1], 2), u[:, i, 1], 2; gas = gas) -
                F_hll(u[:, i, 1], u[:, i, 2], 2; gas = gas)
            )
        )

    # top wall
    @tullio u_next[:, i, end] =
        u[:, i, end] + (
            Δt / Δx * (
                F_hll(u[:, i-1, end], u[:, i, end], 1; gas = gas) -
                F_hll(u[:, i, end], u[:, i+1, end], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, i, end-1], u[:, i, end], 2; gas = gas) -
                F_hll(u[:, i, end], _ghost_u(u[:, i, end], 2), 2; gas = gas)
            )
        )

    # left wall
    @tullio u_next[:, 1, j] =
        u[:, 1, j] + (
            Δt / Δx * (
                F_hll(_ghost_u(u[:, 1, j], 1), u[:, 1, j], 1; gas = gas) -
                F_hll(u[:, 1, j], u[:, 2, j], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, 1, j-1], u[:, 1, j], 2; gas = gas) -
                F_hll(u[:, 1, j], u[:, 1, j+1], 2; gas = gas)
            )
        )

    # right wall
    @tullio u_next[:, end, j] =
        u[:, end, j] + (
            Δt / Δx * (
                F_hll(u[:, end-1, j], u[:, end, j], 1; gas = gas) -
                F_hll(u[:, end, j], _ghost_u(u[:, end, j], 1), 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, 1, j-1], u[:, 1, j], 2; gas = gas) -
                F_hll(u[:, 1, j], u[:, 1, j+1], 2; gas = gas)
            )
        )
end

"""
- `u` is `4` by `nx` by `ny`. i.e. `4x100x100` for a 10,000 grid cell solver in 2D.
"""
function bulk_step!(u_next, u, Δx, Δy, Δt; gas::CaloricallyPerfectGas = DRY_AIR)
    @tullio u_next[:, i, j] =
        u[:, i, j] + (
            Δt / Δx * (
                F_hll(u[:, i-1, j], u[:, i, j], 1; gas = gas) -
                F_hll(u[:, i, j], u[:, i+1, j], 1; gas = gas)
            ) +
            # switch to Δy here eventually
            Δt / Δy * (
                F_hll(u[:, i, j-1], u[:, i, j], 2; gas = gas) -
                F_hll(u[:, i, j], u[:, i, j+1], 2; gas = gas)
            )
        )
    return u_next
end

function maximum_wave_speeds(u; gas::CaloricallyPerfectGas)
    wave_speeds = map(eachslice(u; dims = (2, 3))) do u
        eigenvalues_∇F(u; gas = gas)
    end |> stack
    ax_max =
        max(abs(maximum(wave_speeds[:, 1, :, :])), abs(minimum(wave_speeds[:, 1, :, :])))
    ay_max =
        max(abs(maximum(wave_speeds[:, 2, :, :])), abs(minimum(wave_speeds[:, 2, :, :])))
    return ax_max, ay_max
end

##

function step_euler_hll!(u_next, u, Δx, Δy, Δt; gas::CaloricallyPerfectGas)
    bulk_step!(u_next, u, Δx, Δy, Δt; gas = gas)
    wall_boundaries!(u_next, u, Δx, Δy, Δt; gas = gas)
end

##

free_stream = [1.0, 0.0, 0.0, 100]
uinf = conserved_state_vector(free_stream; gas = DRY_AIR)
u_grid = stack(reshape([uinf for i = 1:100 for j = 1:100], (100, 100)))
ws = stack(map(u -> eigenvalues_∇F(u), eachslice(u_grid; dims = (2, 3))))
F_grid = stack(map(u -> F(u; gas = DRY_AIR), eachslice(u_grid; dims = (2, 3))))

##

function u0(x::Float64, y::Float64)::Vector{Float64}
    if x < 0.5
        return conserved_state_vector([1.225, 2.0, 0.0, 300]; gas=DRY_AIR)
    end
    return conserved_state_vector([1.225, 0.0, -1.0, 300]; gas=DRY_AIR)
end

##

function simulate_euler_2d(
    x_max::Float64,
    y_max::Float64,
    Nx::Int,
    Ny::Int,
    T::Float64,
    u0::Function;
    gas::CaloricallyPerfectGas = DRY_AIR,
    CFL = 0.75,
    write_output = true,
    u_output_file = "data/u.dat",
    t_output_file = "data/t.dat",
    meta_output_file = "data/meta.dat",
)
    write_output && u_io = open(u_output_file; write = true)

    xs = range(0.0, x_max, Nx)
    ys = range(0.0, y_max, Ny)
    u = stack([u0(x, y) for x ∈ xs, y ∈ ys])
    u_next = zeros(eltype(u), size(u))
    t = [0]

    write_output && write(u_io, u)

    while (!(t[end] > T || t[end] ≈ T))
        ax, ay = maximum_wave_speeds(u; gas = gas)
        Δt = CFL * min(Δx / ax, Δy / ay)
        Δt = if t[end] + Δt > T
            Δt = T - t[end]
        end
        step_euler_hll!(u_next, u, Δx, Δy, Δt)
        u = u_next
        push!(t, t[end] + Δt)
        write_output && write(u_io, u)
    end

    if write_output
        open(t_output_file; write = true) do f
            write(f, t)
        end
        open(meta_output_file; write = true) do f
            write(f, length(t))
            for n ∈ size(u)
                write(f, n)
            end
        end
        close(u_io)
    end
end