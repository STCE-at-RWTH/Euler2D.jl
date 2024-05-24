using Euler2D
using ShockwaveProperties
using Unitful

function u0(x)
    uL = ConservedState(PrimitiveState(1.225, [0.0], 595.0); gas = DRY_AIR)
    uR = ConservedState(PrimitiveState(1.225, [0.0], 295.0); gas = DRY_AIR)
    if x < 5.0
        return [1.225, 0.0, 200] #state_to_vector(uL)
    end
    return [1.225, 0.0, 100] #state_to_vector(uR)
end

function u1(x)
    uL = ConservedState(PrimitiveState(2.25, [2.0], 250.); gas = DRY_AIR)
    uR = ConservedState(PrimitiveState(1.225, [0.5], 350.); gas = DRY_AIR)
    if 25. < x < 75.
        return state_to_vector(uL)
    end
    return state_to_vector(uR)
end
##

function simulate_euler_1d(
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

    xs = range(0.0, x_max; length = ncells_x + 1)
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

Δx = 0.005
u = mapreduce(hcat, (0:Δx:10.0)[1:end-1]) do x
    u0(x + Δx / 2)
end

u_next = similar(u)
wrap_bcs = PeriodicAxis()
wall_bcs = WallBoundary(StrongWall(), StrongWall())
Δt = maximum_Δt(wall_bcs, u, Δx, 0.5, 1; gas = DRY_AIR)
step_euler_hll!(u_next, u, Δt, Δx, wall_bcs; gas = DRY_AIR)

simulate_euler_1d(100.0, 2000, wrap_bcs, 0.1, u1; gas = DRY_AIR, CFL = 0.75)

p1 = PrimitiveState(1.25, [0.5], 250.)
c1 = ConservedState(p1; gas=DRY_AIR)
internal_energy_density(c1)
uconvert(u"kPa", pressure(c1; gas=DRY_AIR))
uconvert(u"kPa", pressure(p1; gas=DRY_AIR))
p2 = PrimitiveState(2.5, [0.5], 125.)
c2 = ConservedState(p2; gas=DRY_AIR)
internal_energy_density(c2)
uconvert(u"kPa", pressure(c2; gas=DRY_AIR))
uconvert(u"kPa", pressure(p2; gas=DRY_AIR))

pressure(c2; gas=DRY_AIR) - pressure(c1; gas=DRY_AIR)

##
## NOTES FROM FRIDAY
## WTF is wrong with my boundary conditions?
## maybe I should email Herty / Müller...
## Maybe I should also switch to HLL-C to correct the contact wave rather than smearing out the data.
## hmmmm. 
