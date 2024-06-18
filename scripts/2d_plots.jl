using Euler2D
using LinearAlgebra
using ShockwaveProperties
using Tullio
using Unitful

"""
    simulate_euler_2d(x_bounds, y_bounds, ncells, x_bcs, T, u0; gas, CFL, max_tsteps, write_output, output_tag)

Simulate the solution to the Euler equations from `t=0` to `t=T`, with `u(0, x) = u0(x)`. 
Time step size is computed from the CFL condition.

The simulation will fail if any nonphysical conditions are reached (speed of sound cannot be computed). 
The usual error message for this is printed as "Δt calculation failed" to the command line.

If the simulation is written to disk, two files will be created under `data/`. 
One is the simulation tape, and the other is the full simulation information (`.out`). 
The `.out` file contains, in this order:
- 3 `UInt64`: the dimensions of `u`
- 2 `Float64`: `x0` and `x_max`
- 1 `UInt64`: the number of time steps
- `N_t` `Float64`: The values ``t^k``
- ``N_t⋅N_x⋅N_u`` `Float64`: The values of `u`

Arguments
---
- `x_min`, `x_max`: The x-positions of the left and right boundaries, respectively.
- `ncells_x`: The number of FVM cells in the x-direction.
- `x_bcs`: Boundary conditions on the x-axis. 
- `T`: End time.
- `u0`: ``u(0, x):ℝ^2↦ℝ^3``: conditions at time `t=0`.

Keyword Arguments
---
- `gas=DRY_AIR`: The fluid to be simulated.
- `CFL=0.75`: The CFL condition to apply to `Δt`. Between zero and one, default `0.75`.
- `max_tsteps`: Maximum number of time steps to take. Defaults to "very large".
- `write_output=true`: Should output be written to disk?
- `output_tag="euler_1d"`: File name for the tape and output summary.
"""
function simulate_euler_2d(
    x_bounds,
    y_bounds,
    ncells,
    boundary_conditions,
    T,
    u0;
    gas::CaloricallyPerfectGas = DRY_AIR,
    CFL = 0.75,
    max_tsteps = typemax(Int),
    write_output = true,
    output_tag = "euler_2d",
)
    write_output = write_output && !isempty(output_tag)
    if write_output
        if !isdir("data")
            mkdir("data")
        end
        tape_file = joinpath("data", output_tag * ".tape")
        u_tape = open(tape_file; write = true, read = true, create = true)
    end

    xs = range(x_bounds...; length = ncells[1] + 1)
    Δx = step(xs)
    cell_xs = xs[1:end-1] .+ Δx / 2
    ys = range(y_bounds...; length = ncells[2] + 1)
    Δy = step(ys)
    cell_ys = ys[1:end-1] .+ Δy / 2

    ΔA = (Δx, Δy)

    u = Array{Float64, 3}(undef, 4, ncells...)
    @tullio u[:, i, j] = u0(cell_xs[i], cell_ys[j])
    u_next = zeros(eltype(u), size(u))

    t = [0.0]
    write_output && write(u_tape, u)

    while ((!(t[end] > T || t[end] ≈ T)) && length(t) <= max_tsteps)
        try
            Δt = maximum_Δt(x_bcs, u, Δx, CFL, 1; gas = gas)
        catch err
            @show length(t), t[end]
            println("Δt calculation failed, stopping sim early")
            println(typeof(err))
            break
        end
        if t[end] + Δt > T
            Δt = T - t[end]
        end
        (length(t) % 10 == 0) && @show length(t), t[end], Δt
        step_euler_hll!(u_next, u, Δt, Δx, x_bcs; gas = gas)
        u .= u_next
        push!(t, t[end] + Δt)
        write_output && write(u_tape, u)
    end

    if write_output
        out_file = joinpath("data", output_tag * ".out")
        open(out_file; write = true) do f
            write(f, size(u)...)
            write(f, first(xs), last(xs))
            write(f, length(t))
            write(f, t)
            p = position(u_tape)
            seekstart(u_tape)
            # this could be slow. very slow.
            write(f, read(u_tape))
        end
        close(u_tape)
    end
    return (t[end], u)
end