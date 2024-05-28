using Euler2D
using ShockwaveProperties
using Unitful

##

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
    return (t[end], u_next)
end

##
# SHOCK SCENARIO ONE
# SHOCK AT X = 0
# SUPERSONIC FLOW IMPACTS STATIC ATMOSPHERIC AIR

uL_1 = ConservedState(PrimitiveState(1.225, [2.0], 300.0); gas = DRY_AIR)
uR_1 = ConservedState(PrimitiveState(1.225, [0.0], 350.0); gas = DRY_AIR)

u1(x) = state_to_vector(x < 0 ? uL_1 : uR_1)
left_bc_1 = SupersonicInflow(uL_1)
# convert pressure at outflow to Pascals 
# before stripping units (just to be safe)
right_bc_1 = FixedPressureOutflow(ustrip(u"Pa", pressure(uR_1; gas = DRY_AIR)))
bcs_1 = EdgeBoundary(left_bc_1, right_bc_1)

simulate_euler_1d(
    -50.0,
    50.0,
    8000,
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

uL_2 = ConservedState(PrimitiveState(1.225, [2.0], 300.0); gas = DRY_AIR)
uM_2 = ConservedState(PrimitiveState(1.225, [0.0], 350.0); gas = DRY_AIR)
uR_2 = ConservedState(PrimitiveState(1.225, [-2.0], 300.0); gas = DRY_AIR)

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
    1000,
    bcs_2,
    0.1,
    u2;
    gas = DRY_AIR,
    CFL = 0.75,
    output_tag = "euler_scenario_2",
)

##
## NOTES FROM FRIDAY
## WTF is wrong with my boundary conditions?
## maybe I should email Herty / Müller...
## Maybe I should also switch to HLL-C to correct the contact wave rather than smearing out the data.
## hmmmm. 


vL = state_to_vector(uL_2)
vM = state_to_vector(uM_2)
vR = state_to_vector(uR_2)

s_aL = Euler2D.eigenvalues_∇F(vL, 1; gas=DRY_AIR)
s_aR = Euler2D.eigenvalues_∇F(vM, 1; gas=DRY_AIR)
s_a_roe = Euler2D.roe_matrix_eigenvalues(vL, vM, 1; gas=DRY_AIR)
speeds_a = interface_signal_speeds(vL, vM, 1; gas=DRY_AIR)

s_bL = Euler2D.eigenvalues_∇F(vM, 1; gas=DRY_AIR)
s_bR = Euler2D.eigenvalues_∇F(vR, 1; gas=DRY_AIR)
s_b_roe = Euler2D.roe_matrix_eigenvalues(vM, vR, 1; gas=DRY_AIR)
speeds_b = interface_signal_speeds(vM, vR, 1; gas=DRY_AIR)