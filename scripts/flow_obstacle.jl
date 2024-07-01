using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ambient = launder_units(PrimitiveProps(0.662, (2.5, 0.0), 220.0))
amb2 = launder_units(PrimitiveProps(0.662, (0., 0.), 220.0))

bc_right = SupersonicInflow(ambient, DRY_AIR)
bcs = (bc_right, ExtrapolateToPhantom(), ExtrapolateToPhantom(), ExtrapolateToPhantom() , StrongWall())
bounds = ((-4.0, 4.0), (-4.0, 4.0))
obstacle = [CircularObstacle((0.0, 0.0), 0.75)]
ncells = (250, 250)

Euler2D.simulate_euler_equations_cells(
    0.1,
    bcs,
    obstacle,
    bounds,
    ncells;
    gas = DRY_AIR,
    info_frequency = 25,
    max_tsteps=1000,
    output_tag = "experiment"
) do (x, y)
    x < 0 ? ambient : amb2
end