using ShockwaveProperties
using LinearAlgebra
using Unitful
using Euler2D
using Euler2D: enforce_boundary!, bulk_step!

function get_ics(density_ratio, pressure_ratio)
    ρL = 1.0u"kg/m^3"
    vL = [0.0u"m/s"]
    PL = 10.0u"Pa"
    TL = uconvert(u"K", PL / (ρL * DRY_AIR.R))
    ML = vL / speed_of_sound(ρL, PL; gas = DRY_AIR)

    ρR = density_ratio * ρL
    vR = [0.0u"m/s"]
    PR = pressure_ratio * PL
    TR = uconvert(u"K", PR / (ρR * DRY_AIR.R))
    MR = vR / speed_of_sound(ρR, PR; gas = DRY_AIR)

    s_high = PrimitiveProps(ρL, ML, TL)
    s_low = PrimitiveProps(ρR, MR, TR)
    return (s_high, s_low)
end

s_high, s_low = get_ics(0.125, 0.1)
u0(x) = ConservedProps(x < 0.5 ? s_high : s_low; gas = DRY_AIR) |> state_to_vector
u1(x) = ConservedProps(x > 0.5 ? s_high : s_low; gas = DRY_AIR) |> state_to_vector

copy_bcs = EdgeBoundary(ExtrapolateToPhantom(), ExtrapolateToPhantom())

x_min = 0.0
x_max = 1.0
ncells_x = 400

xs = range(x_min, x_max; length = ncells_x + 1)
Δx = step(xs)

u1_t0 = stack([u0(x + Δx / 2) for x ∈ xs[1:end-1]])
u1_t1 = zeros(eltype(u1_t0), size(u1_t0))
u1_t2 = zeros(eltype(u1_t0), size(u1_t0))

u2_t0 = stack([u1(x + Δx / 2) for x ∈ xs[1:end-1]])
u2_t1 = zeros(eltype(u2_t0), size(u2_t0))
u2_t2 = zeros(eltype(u2_t0), size(u2_t0))

##

t = 0.0
t_end = 0.01
Δt0 = maximum_Δt(copy_bcs, u1_t0, Δx, 0.75, 1; gas = DRY_AIR)

step_euler_hll!(u1_t1, u1_t0, Δt0, Δx, copy_bcs; gas = DRY_AIR);
step_euler_hll!(u2_t1, u2_t0, Δt0, Δx, copy_bcs; gas = DRY_AIR);

Δt1 = maximum_Δt(copy_bcs, u1_t1, Δx, 0.75, 1; gas = DRY_AIR)

step_euler_hll!(u1_t2, u1_t1, Δt1, Δx, copy_bcs; gas = DRY_AIR);
step_euler_hll!(u2_t2, u2_t1, Δt1, Δx, copy_bcs; gas = DRY_AIR);

##

uH = u0(0.0)
uL = u0(1.0)

ϕ_htol = Euler2D.ϕ_hll(uH, uL, 1; gas = DRY_AIR)
ϕ_ltoh = Euler2D.ϕ_hll(uL, uH, 1; gas = DRY_AIR)
ϕ_ltol = Euler2D.ϕ_hll(uL, uL, 1; gas = DRY_AIR)
ϕ_htoh = Euler2D.ϕ_hll(uH, uH, 1; gas = DRY_AIR)

losslr = Δt / Δx * (ϕ_htol - ϕ_htoh) # loss in cell on the left of a right-moving shock
lossrr = Δt / Δx * (ϕ_ltol - ϕ_htol) # loss in cell on the right of a right-moving shock

lossll = Δt / Δx * (ϕ_ltoh - ϕ_ltol) # loss in cell on left of a left-moving shock
lossrl = Δt / Δx * (ϕ_htoh - ϕ_ltoh) # loss in cell on right of a left-moving shock

face_fluxes(u) =
    mapreduce(hcat, @views zip(eachcol(u[:, 1:end-1]), eachcol(u[:, 2:end]))) do (uL, uR)
        Euler2D.ϕ_hll(uL, uR, 1; gas = DRY_AIR)
    end

ff_u1_t0 = face_fluxes(u1_t0)
ff_u2_t0 = face_fluxes(u2_t0)

ff_u1_t1 = face_fluxes(u1_t1)
ff_u2_t1 = face_fluxes(u2_t1)

ff_u1_t2 = face_fluxes(u1_t2)
ff_u2_t2 = face_fluxes(u2_t2)

function verify_face_fluxes(f_right, f_left)
    @assert length(axes(f_right, 1)) % 2 == 1
    @assert length(axes(f_left, 1)) % 2 == 1

    return map(zip(eachcol(f_right), reverse(eachcol(f_left)))) do (f1, f2)
        all(f1 .≈ ([-1, 1, -1] .* f2))
    end
end

function find_separation(face_xs, u0, u0_mirror)
    cell_xs = (face_xs.+step(face_xs)/2)[1:end-1]
    @show cell_xs
    Δx = step(cell_xs)
    u = stack(u0, cell_xs)
    u_mirror = stack(u0_mirror, cell_xs)

    u_next = zeros(eltype(u), size(u))
    u_mirror_next = zeros(eltype(u_mirror), size(u_mirror))

    niter = 0
    t = 0.0
    while niter < 100
        ff = face_fluxes(u)
        ff_mirror = face_fluxes(u_mirror)
        v = verify_face_fluxes(ff, ff_mirror)
        if !all(v)
            println("failed on")
            @show niter, t
            break
        end
        Δt = try
            Δt_regular = maximum_Δt(copy_bcs, u, Δx, 0.75, 1; gas = DRY_AIR)
            Δt_mirror = maximum_Δt(copy_bcs, u_mirror, Δx, 0.75, 1; gas = DRY_AIR)
            if Δt_regular ≉ Δt_mirror
                @show niter, t, Δt_regular, Δt_mirror
            end
            Δt = min(Δt_regular, Δt_mirror)
            Δt
        catch err
            println("Δt calculation failed, aborting")
            println(typeof(err))
            break
        end

        step_euler_hll!(u_next, u, Δt, Δx, copy_bcs; gas = DRY_AIR)
        step_euler_hll!(u_mirror_next, u_mirror, Δt, Δx, copy_bcs; gas = DRY_AIR)

        u = copy(u_next)
        u_mirror = copy(u_mirror_next)
        t += Δt

        niter += 1
    end
    @show niter, t
end