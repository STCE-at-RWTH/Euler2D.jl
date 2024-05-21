using LinearAlgebra
using Tullio
using ShockwaveProperties
using Unitful
using Zygote

##

abstract type BoundaryCondition end
struct StrongWall <: BoundaryCondition end
struct WeakWall <: BoundaryCondition end
struct Periodic <: BoundaryCondition end

##

"""
Computes the numerical flux when the boundary of a cell is a wall.

We know that "weakly" enforcing the boundary condition sets the flux at the wall to `[0 0; P 0; 0 P; 0 0]`.
"""
function F_wall(u, dim; gas::CaloricallyPerfectGas)
    return pressure_u(u; gas = gas) * I[1:length(u), dim+1]
end

"""
Extrapolates the pressure at the wall by using the neighbor cell on the other side.
"""
function F_wall(u1, u2, dim; gas::CaloricallyPerfectGas)
    return (1.5 * pressure_u(u1; gas = gas) - 0.5 * pressure_u(u2; gas = gas)) *
           I[1:length(u), dim+1]
end

function periodic_boundaries!(u_next, u, Δx, Δy, Δt; gas::CaloricallyPerfectGas = DRY_AIR)
    # extrapolate the pressure to the wall and
    # enforce that F̂ along the wall is equal to the pressure
    # and that v⋅n is zero

    # corners
    u_next[:, 1, 1] =
        u[:, 1, 1] + (
            Δt / Δx * (
                F_hll(u[:, end, 1], u[:, 1, 1], 1; gas = gas) -
                F_hll(u[:, 1, 1], u[:, 2, 1], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, 1, end], u[:, 1, 1], 2; gas = gas) -
                F_hll(u[:, 1, 1], u[:, 1, 2], 2; gas = gas)
            )
        )
    u_next[:, 1, end] =
        u[:, 1, end] + (
            Δt / Δx * (
                F_hll(u[:, end, end], u[:, 1, end], 1; gas = gas) -
                F_hll(u[:, 1, end], u[:, 2, end], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, 1, end-1], u[:, 1, end], 2; gas = gas) -
                F_hll(u[:, 1, end], u[:, 1, 1], 2; gas = gas)
            )
        )
    u_next[:, end, 1] =
        u[:, end, 1] + (
            Δt / Δx * (
                F_hll(u[:, end-1, 1], u[:, end, 1], 1; gas = gas) -
                F_hll(u[:, end, 1], u[:, 1, 1], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, end, end], u[:, end, 1], 2; gas = gas) -
                F_hll(u[:, end, 1], u[:, end, 2], 2; gas = gas)
            )
        )
    u_next[:, end, end] =
        u[:, end, end] + (
            Δt / Δx * (
                F_hll(u[:, end-1, end], u[:, end, end], 1; gas = gas) -
                F_hll(u[:, end, end], u[:, 1, end], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, end, end-1], u[:, end, end], 2; gas = gas) -
                F_hll(u[:, end, end], u[:, end, 1], 2; gas = gas)
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
                F_hll(u[:, i, end], u[:, i, 1], 2; gas = gas) -
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
                F_hll(u[:, i, end], u[:, i, 1], 2; gas = gas)
            )
        )

    # left wall
    @tullio u_next[:, 1, j] =
        u[:, 1, j] + (
            Δt / Δx * (
                F_hll(u[:, end, j], u[:, 1, j], 1; gas = gas) -
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
                F_hll(u[:, end, j], u[:, 1, j], 1; gas = gas)
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
function bulk_step!(
    u_next,
    u,
    Δx::Float64,
    Δy::Float64,
    Δt::Float64;
    gas::CaloricallyPerfectGas = DRY_AIR,
)
    @tullio u_next[:, i, j] =
        u[:, i, j] + (
            Δt / Δx * (
                F_hll(u[:, i-1, j], u[:, i, j], 1; gas = gas) -
                F_hll(u[:, i, j], u[:, i+1, j], 1; gas = gas)
            ) +
            Δt / Δy * (
                F_hll(u[:, i, j-1], u[:, i, j], 2; gas = gas) -
                F_hll(u[:, i, j], u[:, i, j+1], 2; gas = gas)
            )
        )
end

function maximum_wave_speeds(u; gas::CaloricallyPerfectGas)
    wave_speeds = map(eachslice(u; dims = (2, 3))) do u
        eigenvalues_∇F(u, 1:2; gas = gas)
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
    periodic_boundaries!(u_next, u, Δx, Δy, Δt; gas = gas)
end

##

free_stream = [4.0, 3.0, 1.0, 100]
uinf = conserved_state_vector(free_stream; gas = DRY_AIR)
u_grid = stack(reshape([uinf for i = 1:100 for j = 1:100], (100, 100)))
ws = stack(map(u -> eigenvalues_∇F(u, 2:3), eachslice(u_grid; dims = (2, 3))))
F_grid = stack(map(u -> F(u; gas = DRY_AIR), eachslice(u_grid; dims = (2, 3))))

##

function u0(x::Float64, y::Float64)::Vector{Float64}
    if x < 0.5
        return conserved_state_vector([2.225, 0.0, 0.0, 300]; gas = DRY_AIR)
    end
    return conserved_state_vector([2.225, 0.0, 0.0, 300]; gas = DRY_AIR)
end

##

function simulate_euler_2d(
    x_max::Float64,
    y_max::Float64,
    ncells_x::Int,
    ncells_y::Int,
    T::Float64,
    u0::Function;
    gas::CaloricallyPerfectGas = DRY_AIR,
    CFL = 0.75,
    max_tsteps = typemax(Int),
    write_output = true,
    u_output_file = "data/u.dat",
    t_output_file = "data/t.dat",
    meta_output_file = "data/meta.dat",
)
    write_output && (u_io = open(u_output_file; write = true))

    xs = range(0.0, x_max; length = ncells_x + 1)
    ys = range(0.0, y_max; length = ncells_y + 1)
    Δx = step(xs)
    Δy = step(ys)
    u = stack([u0(x + Δx / 2, y + Δy / 2) for x ∈ xs[1:end-1], y ∈ ys[1:end-1]])
    u_next = zeros(eltype(u), size(u))
    t = [0.0]

    write_output && write(u_io, u)

    while ((!(t[end] > T || t[end] ≈ T)) && length(t) <= max_tsteps)
        ax, ay = maximum_wave_speeds(u; gas = gas)
        Δt = CFL * min(Δx / ax, Δy / ay)
        if t[end] + Δt > T
            Δt = T - t[end]
        end
        @show length(t), t[end], Δt
        step_euler_hll!(u_next, u, Δx, Δy, Δt; gas = gas)
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
    return (t[end], u_next)
end
##

# (a, b) = simulate_euler_2d(100.0, 100.0, 50, 50, 1.0, u0; max_tsteps = 10000)