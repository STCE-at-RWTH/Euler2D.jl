using Euler2D
using LinearAlgebra
using ShockwaveProperties
using Unitful


"""
    simulate_euler_1d(x_min, x_max, ncells_x, x_bcs, T, u0; gas, CFL, max_tsteps, write_output, output_tag)

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
- `u0`: ``u(0, x):ℝ↦ℝ^3``: conditions at time `t=0`.

Keyword Arguments
---
- `gas=DRY_AIR`: The fluid to be simulated.
- `CFL=0.75`: The CFL condition to apply to `Δt`. Between zero and one, default `0.75`.
- `max_tsteps`: Maximum number of time steps to take. Defaults to "very large".
- `write_output=true`: Should output be written to disk?
- `output_tag="euler_1d"`: File name for the tape and output summary.
"""
function simulate_euler_1d(
    x_min::Float64,
    x_max::Float64,
    ncells_x::Int,
    x_bcs::BoundaryCondition,
    T::Float64,
    u0::Function;
    gas::CaloricallyPerfectGas = DRY_AIR,
    CFL = 0.75,
    max_tsteps = typemax(Int),
    write_output = true,
    output_tag = "euler_1d",
)
    write_output = write_output && !isempty(output_tag)
    if write_output
        if !isdir("data")
            mkdir("data")
        end
        tape_file = joinpath("data", output_tag * ".tape")
        u_tape = open(tape_file; write = true, read = true, create = true)
    end

    xs = range(x_min, x_max; length = ncells_x + 1)
    Δx = step(xs)
    u = stack([u0(x + Δx / 2) for x ∈ xs[1:end-1]])
    u_next = zeros(eltype(u), size(u))
    t = [0.0]

    write_output && write(u_tape, u)

    while ((!(t[end] > T || t[end] ≈ T)) && length(t) <= max_tsteps)
        try
            Δt = maximum_Δt(x_bcs, u, Δx, CFL, 1; gas = gas)
        catch err
            @show length(t), t[end]
            println("Δt calculation failed.")
            println(typeof(err))
            break
        end
        if t[end] + Δt > T
            Δt = T - t[end]
        end
        (length(t) % 10 == 0) && @show length(t), t[end], Δt
        step_euler_hll!(u_next, u, Δt, Δx, x_bcs; gas = gas)
        u = u_next
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

##
# SHOCK SCENARIO ONE
# SHOCK AT X = 0
# SUPERSONIC FLOW IMPACTS STATIC ATMOSPHERIC AIR

uL_1 = ConservedProps(PrimitiveProps(1.225, [1.5], 300.0); gas = DRY_AIR)
uR_1 = ConservedProps(PrimitiveProps(1.225, [0.0], 350.0); gas = DRY_AIR)

u1(x) = state_to_vector(x < 0 ? uL_1 : uR_1)
left_bc_1 = SupersonicInflow(uL_1)
# convert pressure at outflow to Pascals 
# before stripping units (just to be safe)
# right_bc_1 = FixedPressureOutflow(ustrip(u"Pa", pressure(uR_1; gas = DRY_AIR)))
right_bc_1 = FixedPhantomOutside(uR_1)
bcs_1 = EdgeBoundary(left_bc_1, right_bc_1)

# use 1001 cells to avoid a grid-aligned shock...
simulate_euler_1d(
    -25.0,
    225.0,
    500,
    bcs_1,
    0.1,
    u1;
    gas = DRY_AIR,
    CFL = 0.75,
    output_tag = "euler_scenario_1",
)

##
# SHOCK SCENARIO TWO
# SHOCKS AT X = -50 and X = 50
# SUPERSONIC INFLOW ON BOTH SIDES

uL_2 = ConservedProps(PrimitiveProps(1.225, [2.0], 300.0); gas = DRY_AIR)
uM_2 = ConservedProps(PrimitiveProps(1.225, [0.0], 350.0); gas = DRY_AIR)
uR_2 = ConservedProps(PrimitiveProps(1.225, [-2.0], 300.0); gas = DRY_AIR)

interface_signal_speeds(state_to_vector(uL_2), state_to_vector(uM_2), 1; gas=DRY_AIR)
Euler2D.ϕ_hll(state_to_vector(uL_2), state_to_vector(uM_2), 1; gas=DRY_AIR)


function u2(x)
    res = if x < -50
        uL_2
    elseif x > 50
        uR_2
    else
        uM_2
    end
    return state_to_vector(res)
end

left_bc_2 = SupersonicInflow(uL_2)
right_bc_2 = SupersonicInflow(uR_2)
bcs_2 = EdgeBoundary(left_bc_2, right_bc_2)

simulate_euler_1d(
    -75.0,
    75.0,
    2500,
    bcs_2,
    0.15,
    u2;
    gas = DRY_AIR,
    CFL = 0.75,
    output_tag = "euler_scenario_2",
)

## SHOCK SCENARIO 3
# SOD SHOCK TUBE 1

ρL = 1.0u"kg/m^3"
vL = [0.0u"m/s"]
PL = 10.0u"Pa"
TL = uconvert(u"K", PL / (ρL * DRY_AIR.R))
ML = vL/speed_of_sound(ρL, PL; gas=DRY_AIR)

ρR = 0.125 * ρL
vR = [0.0u"m/s"]
PR = 0.1 * PL
TR = uconvert(u"K", PR/(ρR * DRY_AIR.R))
MR = vR/speed_of_sound(ρR, PR; gas=DRY_AIR)

s_high = PrimitiveProps(ρL, ML, TL)
s_low = PrimitiveProps(ρR, MR, TR)

sod1_bcs = EdgeBoundary(FixedPhantomOutside(s_high, DRY_AIR), FixedPhantomOutside(s_low, DRY_AIR))
copy_bcs = EdgeBoundary(ExtrapolateToPhantom(), ExtrapolateToPhantom())
u0_sod1(x) = ConservedProps(x < 0.5 ? s_high : s_low; gas=DRY_AIR) |> state_to_vector

simulate_euler_1d(0.0, 2.0, 2000, copy_bcs, 0.05, u0_sod1; gas=DRY_AIR, CFL=0.75, output_tag="sod1")

# SOD SCENARIO 2

u0_sod2(x) = ConservedProps(x < 1.5 ? s_low : s_high; gas=DRY_AIR) |> state_to_vector
simulate_euler_1d(0.0, 2.0, 2000, copy_bcs, 0.05, u0_sod2; gas=DRY_AIR, CFL=0.75, output_tag="sod2")